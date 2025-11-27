using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.Data.Sqlite;
using Dapper;
using DotaPredictor.Services;
using System.Text.Json;
using System.Collections.Concurrent;

var builder = WebApplication.CreateBuilder(args);

// --- SERVICES ---
builder.Services.AddHostedService<FetcherService>();
builder.Services.AddHttpClient();
// Реєструємо ModelEngine, система сама передасть туди Logger
builder.Services.AddSingleton<ModelEngine>();

// Налаштування логування (щоб було видно в консолі)
builder.Logging.ClearProviders();
builder.Logging.AddConsole();

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

InitDatabase();

// --- API ---

app.MapGet("/api/status", async (ModelEngine engine) => 
{
    using var connection = new SqliteConnection("Data Source=dota_data.db");
    // Оптимізація: Count(*) може бути повільним на великих даних в SQLite, 
    // але для <1млн це ок.
    var count = await connection.ExecuteScalarAsync<int>("SELECT COUNT(*) FROM Matches");
    
    return Results.Json(new { 
        totalMatches = count, 
        isRunning = !FetcherService.IsPaused,
        // Повертаємо логи, які тепер включатимуть і тренування
        logs = FetcherService.Logs.ToArray(),
        training = engine.GetState()
    });
});

app.MapPost("/api/fetcher/start", () => { FetcherService.IsPaused = false; return Results.Ok(); });
app.MapPost("/api/fetcher/stop", () => { FetcherService.IsPaused = true; return Results.Ok(); });

app.MapPost("/api/train", (ModelEngine engine) => 
{
    if (engine.GetState().IsTraining) return Results.BadRequest("Busy");
    // Запуск в окремому потоці, щоб не блокувати API
    Task.Run(() => engine.Train());
    return Results.Ok("Started");
});

app.MapPost("/api/predict", (PredictionRequest req, ModelEngine engine) => 
{
    var insight = engine.PredictWithInsights(req);
    
    string reasonText = "";
    if (insight.Probability > 0.65) reasonText = "Radiant has a dominant strategic advantage.";
    else if (insight.Probability > 0.53) reasonText = "Radiant slight statistical edge.";
    else if (insight.Probability < 0.35) reasonText = "Dire has a dominant strategic advantage.";
    else if (insight.Probability < 0.47) reasonText = "Dire slight statistical edge.";
    else reasonText = "Dead even. Skill matchup.";

    if (insight.Factors.Any())
    {
        var topFactor = insight.Factors.OrderByDescending(f => Math.Abs(f.Impact)).First();
        reasonText += $" Key: {topFactor.HeroId} ({(topFactor.Impact > 0 ? "+" : "")}{topFactor.Impact:P1})";
    }

    return Results.Json(new { 
        radiantWinProbability = insight.Probability,
        reason = reasonText,
        factors = insight.Factors
    });
});

app.Run();

void InitDatabase()
{
    using var connection = new SqliteConnection("Data Source=dota_data.db");
    connection.Open();
    connection.Execute(@"
        CREATE TABLE IF NOT EXISTS Matches (
            match_id INTEGER PRIMARY KEY, radiant_win BOOLEAN, start_time INTEGER,
            duration INTEGER, avg_rank_tier INTEGER, radiant_team TEXT, dire_team TEXT, raw_json TEXT
        )
    ");
}

// --- DATA MODELS ---

public class ModelInput
{
    [LoadColumn(0)] public float Rank { get; set; }
    [LoadColumn(1, 280)] [VectorType(280)] public float[]? Heroes { get; set; } 
    [LoadColumn(281, 290)] [VectorType(10)] public float[]? Lanes { get; set; }
    [LoadColumn(291)] public float SynergyRadiant { get; set; }
    [LoadColumn(292)] public float SynergyDire { get; set; }
    [LoadColumn(293)] public float CounterScore { get; set; }
    [LoadColumn(294)] public bool Label { get; set; }
}

public class ModelOutput
{
    [ColumnName("PredictedLabel")] public bool Prediction { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}

public class PredictionRequest
{
    public int Rank { get; set; }
    public int[] Radiant { get; set; } = Array.Empty<int>();
    public int[] Dire { get; set; } = Array.Empty<int>();
    public int[] RadiantLanes { get; set; } = Array.Empty<int>();
    public int[] DireLanes { get; set; } = Array.Empty<int>();
}

public class InfluenceFactor
{
    public int HeroId { get; set; }
    public bool IsRadiant { get; set; }
    public float Impact { get; set; }
}

public class PredictionResult
{
    public float Probability { get; set; }
    public List<InfluenceFactor> Factors { get; set; } = new();
}

public class TrainingState
{
    public bool IsTraining { get; set; }
    public int ProgressPercent { get; set; }
    public string Status { get; set; } = "Idle";
    public int LastTrainedOnCount { get; set; }
    public double LastAccuracy { get; set; }
}

// --- KNOWLEDGE BASE ---
public class KnowledgeBase
{
    public Dictionary<string, float> SynergyMap { get; set; } = new();
    public Dictionary<string, float> CounterMap { get; set; } = new();

    public void Build(List<(int[] r, int[] d, bool win)> matches)
    {
        var pairWins = new ConcurrentDictionary<string, int>();
        var pairTotal = new ConcurrentDictionary<string, int>();
        var counterWins = new ConcurrentDictionary<string, int>();
        var counterTotal = new ConcurrentDictionary<string, int>();

        Parallel.ForEach(matches, match => 
        {
            // Synergy
            for (int i = 0; i < match.r.Length; i++)
                for (int j = i + 1; j < match.r.Length; j++)
                {
                    UpdatePair(match.r[i], match.r[j], match.win, pairWins, pairTotal);
                }

            for (int i = 0; i < match.d.Length; i++)
                for (int j = i + 1; j < match.d.Length; j++)
                {
                    UpdatePair(match.d[i], match.d[j], !match.win, pairWins, pairTotal);
                }

            // Counter
            foreach (var rHero in match.r)
                foreach (var dHero in match.d)
                {
                    string key = $"{rHero}|{dHero}";
                    counterTotal.AddOrUpdate(key, 1, (k, v) => v + 1);
                    if (match.win) counterWins.AddOrUpdate(key, 1, (k, v) => v + 1);
                }
        });

        SynergyMap.Clear();
        foreach (var kvp in pairTotal)
        {
            if (kvp.Value < 5) continue; 
            float wr = (float)pairWins.GetValueOrDefault(kvp.Key, 0) / kvp.Value;
            SynergyMap[kvp.Key] = wr - 0.5f; 
        }

        CounterMap.Clear();
        foreach (var kvp in counterTotal)
        {
            if (kvp.Value < 5) continue;
            float wr = (float)counterWins.GetValueOrDefault(kvp.Key, 0) / kvp.Value;
            CounterMap[kvp.Key] = wr - 0.5f;
        }
    }

    private void UpdatePair(int h1, int h2, bool win, ConcurrentDictionary<string, int> wins, ConcurrentDictionary<string, int> total)
    {
        int min = Math.Min(h1, h2);
        int max = Math.Max(h1, h2);
        string key = $"{min}|{max}";
        total.AddOrUpdate(key, 1, (k, v) => v + 1);
        if (win) wins.AddOrUpdate(key, 1, (k, v) => v + 1);
    }

    public float CalculateTeamSynergy(int[] team)
    {
        float score = 0;
        for (int i = 0; i < team.Length; i++)
            for (int j = i + 1; j < team.Length; j++)
            {
                int min = Math.Min(team[i], team[j]);
                int max = Math.Max(team[i], team[j]);
                if (SynergyMap.TryGetValue($"{min}|{max}", out float s)) score += s;
            }
        return score;
    }

    public float CalculateCounterScore(int[] rad, int[] dire)
    {
        float score = 0;
        foreach (var r in rad)
            foreach (var d in dire)
            {
                if (CounterMap.TryGetValue($"{r}|{d}", out float s)) score += s;
            }
        return score;
    }
}

// --- ML ENGINE ---

public class ModelEngine
{
    private readonly string _modelPath = "dota_model.zip";
    private readonly string _kbPath = "knowledge_base.json";
    
    // LOGGER: Для виводу в консоль
    private readonly ILogger<ModelEngine> _logger;
    
    private MLContext _mlContext;
    private ITransformer? _trainedModel;
    private PredictionEngine<ModelInput, ModelOutput>? _predEngine;
    private TrainingState _state = new();
    private KnowledgeBase _kb = new();
    private const int TOTAL_HERO_ID_SLOTS = 140; 

    // Constructor Injection: Отримуємо логер автоматично
    public ModelEngine(ILogger<ModelEngine> logger)
    {
        _logger = logger;
        _mlContext = new MLContext(seed: 42);
        LoadModel();
    }

    public TrainingState GetState() => _state;

    // Helper: Пишемо і в консоль, і в чергу логів для UI
    private void LogInfo(string message)
    {
        // 1. У консоль (Термінал)
        _logger.LogInformation(message);
        
        // 2. У чергу для UI (щоб бачити на сайті)
        var time = DateTime.Now.ToString("HH:mm:ss");
        FetcherService.Logs.Enqueue($"[{time}] [AI] {message}");
        
        // Чистимо старі логи
        if (FetcherService.Logs.Count > 100) FetcherService.Logs.TryDequeue(out _);
        
        // 3. Оновлюємо статус
        _state.Status = message;
    }

    private void LoadModel()
    {
        if (File.Exists(_modelPath))
        {
            try 
            {
                _trainedModel = _mlContext.Model.Load(_modelPath, out var schema);
                _predEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_trainedModel);
                
                if (File.Exists(_modelPath + ".meta"))
                {
                    var meta = File.ReadAllText(_modelPath + ".meta").Split('|');
                    if (meta.Length >= 1 && int.TryParse(meta[0], out int count)) _state.LastTrainedOnCount = count;
                    if (meta.Length >= 2 && double.TryParse(meta[1], out double acc)) _state.LastAccuracy = acc;
                }

                if (File.Exists(_kbPath))
                {
                    var json = File.ReadAllText(_kbPath);
                    _kb = JsonSerializer.Deserialize<KnowledgeBase>(json) ?? new KnowledgeBase();
                }
                LogInfo($"Model loaded. Trained on {_state.LastTrainedOnCount} matches.");
            }
            catch (Exception ex) { 
                _logger.LogError(ex, "Failed to load model"); 
            }
        }
    }

    public PredictionResult PredictWithInsights(PredictionRequest req)
    {
        if (_predEngine == null) return new PredictionResult { Probability = 0.5f };

        var baseInput = CreateInput(req);
        var basePrediction = _predEngine.Predict(baseInput);
        float baseProb = basePrediction.Probability;

        var result = new PredictionResult { Probability = baseProb };

        // XAI
        foreach (var heroId in req.Radiant)
        {
            if (heroId <= 0) continue;
            var modReq = CloneRequest(req);
            modReq.Radiant = modReq.Radiant.Where(id => id != heroId).ToArray(); 
            var modProb = _predEngine.Predict(CreateInput(modReq)).Probability;
            result.Factors.Add(new InfluenceFactor { HeroId = heroId, IsRadiant = true, Impact = baseProb - modProb });
        }

        foreach (var heroId in req.Dire)
        {
            if (heroId <= 0) continue;
            var modReq = CloneRequest(req);
            modReq.Dire = modReq.Dire.Where(id => id != heroId).ToArray();
            var modProb = _predEngine.Predict(CreateInput(modReq)).Probability;
            result.Factors.Add(new InfluenceFactor { HeroId = heroId, IsRadiant = false, Impact = modProb - baseProb });
        }

        result.Factors = result.Factors.OrderByDescending(f => Math.Abs(f.Impact)).ToList();
        return result;
    }

    private PredictionRequest CloneRequest(PredictionRequest original)
    {
        return new PredictionRequest {
            Rank = original.Rank,
            Radiant = (int[])original.Radiant.Clone(),
            Dire = (int[])original.Dire.Clone(),
            RadiantLanes = (int[])original.RadiantLanes.Clone(),
            DireLanes = (int[])original.DireLanes.Clone()
        };
    }

    private ModelInput CreateInput(PredictionRequest req)
    {
        var input = new ModelInput
        {
            Rank = req.Rank * 10, 
            Heroes = new float[TOTAL_HERO_ID_SLOTS * 2],
            Lanes = new float[10],
            SynergyRadiant = _kb.CalculateTeamSynergy(req.Radiant),
            SynergyDire = _kb.CalculateTeamSynergy(req.Dire),
            CounterScore = _kb.CalculateCounterScore(req.Radiant, req.Dire)
        };

        foreach (var id in req.Radiant) if (id > 0 && id < TOTAL_HERO_ID_SLOTS) input.Heroes![id] = 1;
        foreach (var id in req.Dire)    if (id > 0 && id < TOTAL_HERO_ID_SLOTS) input.Heroes![TOTAL_HERO_ID_SLOTS + id] = 1;

        for (int i = 0; i < req.RadiantLanes.Length && i < 5; i++) input.Lanes![i] = req.RadiantLanes[i];
        for (int i = 0; i < req.DireLanes.Length && i < 5; i++) input.Lanes![5 + i] = req.DireLanes[i];

        return input;
    }

    // Legacy
    public ModelOutput Predict(PredictionRequest req) => _predEngine?.Predict(CreateInput(req)) ?? new ModelOutput();

    public void Train()
    {
        try
        {
            _state.IsTraining = true;
            _state.ProgressPercent = 0;
            LogInfo("Starting training sequence...");
            
            LogInfo("Fetching matches from DB...");
            using var connection = new SqliteConnection("Data Source=dota_data.db");
            var rawData = connection.Query<(int rank, string rTeam, string dTeam, bool win)>("SELECT avg_rank_tier, radiant_team, dire_team, radiant_win FROM Matches").ToList();
            
            if (rawData.Count < 50)
            {
                LogInfo($"Error: Not enough data ({rawData.Count} matches). Need 50+.");
                _state.IsTraining = false;
                return;
            }

            // STEP 1: KB
            _state.ProgressPercent = 10;
            LogInfo($"Analyzing Meta & Synergies on {rawData.Count} matches...");
            
            var parsedMatches = new List<(int[] r, int[] d, bool win)>();
            foreach(var m in rawData) {
                var r = JsonSerializer.Deserialize<int[]>(m.rTeam);
                var d = JsonSerializer.Deserialize<int[]>(m.dTeam);
                if(r != null && d != null) parsedMatches.Add((r, d, m.win));
            }

            _kb.Build(parsedMatches);
            File.WriteAllText(_kbPath, JsonSerializer.Serialize(_kb));
            LogInfo("Knowledge Base built.");

            // STEP 2: VECTORS
            _state.ProgressPercent = 30;
            LogInfo("Vectorizing features (One-Hot + Engineered)...");

            var trainingData = new List<ModelInput>();
            int idx = 0;
            foreach (var match in rawData)
            {
                var pm = parsedMatches[idx++];
                
                var input = new ModelInput 
                { 
                    Rank = match.rank, 
                    Label = match.win, 
                    Heroes = new float[TOTAL_HERO_ID_SLOTS * 2],
                    Lanes = new float[10],
                    SynergyRadiant = _kb.CalculateTeamSynergy(pm.r),
                    SynergyDire = _kb.CalculateTeamSynergy(pm.d),
                    CounterScore = _kb.CalculateCounterScore(pm.r, pm.d)
                };

                foreach (var id in pm.r) if (id > 0 && id < TOTAL_HERO_ID_SLOTS) input.Heroes![id] = 1;
                foreach (var id in pm.d) if (id > 0 && id < TOTAL_HERO_ID_SLOTS) input.Heroes![TOTAL_HERO_ID_SLOTS + id] = 1;
                
                trainingData.Add(input);
            }

            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            var split = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.15); 

            _state.ProgressPercent = 50;
            LogInfo("Training LightGBM (Gradient Boosting)...");

            // OPTIONS
            var options = new LightGbmBinaryTrainer.Options
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                NumberOfLeaves = 50,
                MinimumExampleCountPerLeaf = 20,
                LearningRate = 0.02,
                NumberOfIterations = 1500
                // Прибрали проблемні поля Booster для сумісності з консольним запуском
            };

            var pipeline = _mlContext.Transforms.Concatenate("Features", 
                    "Rank", "Heroes", "Lanes", 
                    "SynergyRadiant", "SynergyDire", "CounterScore")
                .Append(_mlContext.BinaryClassification.Trainers.LightGbm(options));

            _trainedModel = pipeline.Fit(split.TrainSet);
            
            _state.ProgressPercent = 80;
            LogInfo("Evaluating model accuracy...");

            var predictions = _trainedModel.Transform(split.TestSet);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions, "Label");

            _state.ProgressPercent = 90;
            LogInfo($"Validation Accuracy: {metrics.Accuracy:P2}. Saving...");

            _mlContext.Model.Save(_trainedModel, dataView.Schema, _modelPath);
            File.WriteAllText(_modelPath + ".meta", $"{rawData.Count}|{metrics.Accuracy}");
            
            _predEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_trainedModel);

            _state.LastTrainedOnCount = rawData.Count;
            _state.LastAccuracy = metrics.Accuracy;
            LogInfo("Training Complete. Model Ready.");
            _state.Status = $"Ready (Acc: {metrics.Accuracy:P1})";
            _state.ProgressPercent = 100;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Training crash");
            LogInfo("CRITICAL ERROR during training. Check console.");
            _state.Status = "Training Failed";
        }
        finally
        {
            _state.IsTraining = false;
        }
    }
}