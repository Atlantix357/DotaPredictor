using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.Data.Sqlite;
using Dapper;
using DotaPredictor.Services;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

// --- SERVICES ---
builder.Services.AddHostedService<FetcherService>();
builder.Services.AddHttpClient();
builder.Services.AddSingleton<ModelEngine>();

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

InitDatabase();

// --- API ---

app.MapGet("/api/status", async (ModelEngine engine) => 
{
    using var connection = new SqliteConnection("Data Source=dota_data.db");
    var count = await connection.ExecuteScalarAsync<int>("SELECT COUNT(*) FROM Matches");
    
    return Results.Json(new { 
        totalMatches = count, 
        isRunning = !FetcherService.IsPaused,
        logs = FetcherService.Logs.ToArray(),
        training = engine.GetState()
    });
});

app.MapPost("/api/fetcher/start", () => { FetcherService.IsPaused = false; return Results.Ok(); });
app.MapPost("/api/fetcher/stop", () => { FetcherService.IsPaused = true; return Results.Ok(); });

app.MapPost("/api/train", (ModelEngine engine) => 
{
    if (engine.GetState().IsTraining) return Results.BadRequest("Busy");
    Task.Run(() => engine.Train());
    return Results.Ok("Started");
});

app.MapPost("/api/predict", (PredictionRequest req, ModelEngine engine) => 
{
    var insight = engine.PredictWithInsights(req);
    
    string reasonText = "";
    if (insight.Probability > 0.65) reasonText = "Radiant has a dominant draft advantage.";
    else if (insight.Probability > 0.53) reasonText = "Radiant slight statistical edge.";
    else if (insight.Probability < 0.35) reasonText = "Dire has a dominant draft advantage.";
    else if (insight.Probability < 0.47) reasonText = "Dire slight statistical edge.";
    else reasonText = "Dead even. Pure skill match.";

    if (insight.Factors.Any())
    {
        var topFactor = insight.Factors.OrderByDescending(f => Math.Abs(f.Impact)).First();
        reasonText += $" Key factor: {topFactor.HeroId} ({(topFactor.Impact > 0 ? "+" : "")}{topFactor.Impact:P1})";
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
    
    // Вектор героїв (140 Radiant + 140 Dire)
    [LoadColumn(1, 280)] [VectorType(280)] public float[]? Heroes { get; set; } 
    
    // Вектор ліній (5 Radiant + 5 Dire) - значення 0-4 (Unknown, Safe, Mid, Off, Roam)
    [LoadColumn(281, 290)] [VectorType(10)] public float[]? Lanes { get; set; }

    [LoadColumn(291)] public bool Label { get; set; }
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

// --- ML ENGINE ---

public class ModelEngine
{
    private readonly string _modelPath = "dota_model.zip";
    private MLContext _mlContext;
    private ITransformer? _trainedModel;
    private PredictionEngine<ModelInput, ModelOutput>? _predEngine;
    private TrainingState _state = new();
    private const int TOTAL_HERO_ID_SLOTS = 140; 

    public ModelEngine()
    {
        _mlContext = new MLContext(seed: 42);
        LoadModel();
    }

    public TrainingState GetState() => _state;

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
            }
            catch { Console.WriteLine("Model corrupted."); }
        }
    }

    public PredictionResult PredictWithInsights(PredictionRequest req)
    {
        if (_predEngine == null) return new PredictionResult { Probability = 0.5f };

        var baseInput = CreateInput(req);
        var basePrediction = _predEngine.Predict(baseInput);
        float baseProb = basePrediction.Probability;

        var result = new PredictionResult { Probability = baseProb };

        // Аналіз впливу (Perturbation Analysis)
        foreach (var heroId in req.Radiant)
        {
            if (heroId <= 0) continue;
            var modReq = CloneRequest(req);
            modReq.Radiant = modReq.Radiant.Where(id => id != heroId).ToArray(); // Прибираємо героя
            
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

    // Helper to clone request for simulation
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
            // FIX: Масштабування рангу. UI дає 1-8, OpenDota дає 10-80. 
            Rank = req.Rank * 10, 
            Heroes = new float[TOTAL_HERO_ID_SLOTS * 2],
            Lanes = new float[10]
        };

        // Заповнюємо героїв
        foreach (var id in req.Radiant) if (id > 0 && id < TOTAL_HERO_ID_SLOTS) input.Heroes![id] = 1;
        foreach (var id in req.Dire)    if (id > 0 && id < TOTAL_HERO_ID_SLOTS) input.Heroes![TOTAL_HERO_ID_SLOTS + id] = 1;

        // FIX: Заповнюємо лінії (якщо вони передані)
        // Перші 5 - Radiant, наступні 5 - Dire
        for (int i = 0; i < req.RadiantLanes.Length && i < 5; i++) input.Lanes![i] = req.RadiantLanes[i];
        for (int i = 0; i < req.DireLanes.Length && i < 5; i++) input.Lanes![5 + i] = req.DireLanes[i];

        return input;
    }

    // Старий метод для сумісності
    public ModelOutput Predict(PredictionRequest req) => _predEngine?.Predict(CreateInput(req)) ?? new ModelOutput();

    public void Train()
    {
        try
        {
            _state.IsTraining = true;
            _state.ProgressPercent = 0;
            _state.Status = "Fetching matches from DB...";
            
            using var connection = new SqliteConnection("Data Source=dota_data.db");
            var rawData = connection.Query<(int rank, string rTeam, string dTeam, bool win)>("SELECT avg_rank_tier, radiant_team, dire_team, radiant_win FROM Matches").ToList();
            
            if (rawData.Count < 50)
            {
                _state.Status = "Error: Need >50 matches";
                _state.IsTraining = false;
                return;
            }

            _state.ProgressPercent = 10;
            _state.Status = $"Vectorizing {rawData.Count} matches...";

            var trainingData = new List<ModelInput>();
            foreach (var match in rawData)
            {
                var input = new ModelInput 
                { 
                    Rank = match.rank, // Тут з бази вже приходить 10-80
                    Label = match.win, 
                    Heroes = new float[TOTAL_HERO_ID_SLOTS * 2],
                    Lanes = new float[10] // Ініціалізуємо нулями, бо історичних даних про лінії часто немає
                };

                try {
                    var rIds = JsonSerializer.Deserialize<int[]>(match.rTeam);
                    var dIds = JsonSerializer.Deserialize<int[]>(match.dTeam);
                    if (rIds != null) foreach (var id in rIds) if (id > 0 && id < TOTAL_HERO_ID_SLOTS) input.Heroes![id] = 1;
                    if (dIds != null) foreach (var id in dIds) if (id > 0 && id < TOTAL_HERO_ID_SLOTS) input.Heroes![TOTAL_HERO_ID_SLOTS + id] = 1;
                    trainingData.Add(input);
                } catch { }
            }

            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            var split = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.15); 

            _state.ProgressPercent = 30;
            _state.Status = "Training LightGBM (Pro)...";

            // --- PRO SETTINGS (IMPLEMENTED CORRECTLY) ---
            var options = new LightGbmBinaryTrainer.Options
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                
                // Основні параметри
                NumberOfLeaves = 50,
                MinimumExampleCountPerLeaf = 20,
                LearningRate = 0.02,
                NumberOfIterations = 1500,

                // FIX: Глибинні налаштування через Booster
                Booster = new GradientBooster.Options
                {
                    L2Regularization = 0.5,      // Штраф за перенавчання
                    FeatureFraction = 0.9,       // 90% фіч на дерево
                    SubsampleFraction = 0.8,     // Беггінг (80% даних)
                    SubsampleFrequency = 5
                }
            };

            // Додаємо Lanes у список Features
            var pipeline = _mlContext.Transforms.Concatenate("Features", "Rank", "Heroes", "Lanes")
                .Append(_mlContext.BinaryClassification.Trainers.LightGbm(options));

            _trainedModel = pipeline.Fit(split.TrainSet);
            
            _state.ProgressPercent = 80;
            _state.Status = "Evaluating accuracy...";

            var predictions = _trainedModel.Transform(split.TestSet);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions, "Label");

            _state.ProgressPercent = 90;
            _state.Status = $"Saving (Acc: {metrics.Accuracy:P2})..."; 

            _mlContext.Model.Save(_trainedModel, dataView.Schema, _modelPath);
            File.WriteAllText(_modelPath + ".meta", $"{rawData.Count}|{metrics.Accuracy}");
            
            _predEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_trainedModel);

            _state.LastTrainedOnCount = rawData.Count;
            _state.LastAccuracy = metrics.Accuracy;
            _state.Status = $"Ready (Acc: {metrics.Accuracy:P1})";
            _state.ProgressPercent = 100;
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex);
            _state.Status = "Training Failed (See Console)";
        }
        finally
        {
            _state.IsTraining = false;
        }
    }
}