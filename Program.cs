using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.Data.Sqlite;
using Dapper;
using DotaPredictor.Services;
using System.Text.Json;
using System.Collections.Concurrent;
using System.Globalization;

var builder = WebApplication.CreateBuilder(args);

// --- SERVICES ---
builder.Services.AddHostedService<FetcherService>();
builder.Services.AddHttpClient();
builder.Services.AddSingleton<ModelEngine>();

// Використовуємо стандартний JSON (camelCase)
builder.Logging.ClearProviders();
builder.Logging.AddConsole();

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

InitDatabase();

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

// --- API ---

app.MapGet("/api/status", async (ModelEngine engine) => 
{
    using var connection = new SqliteConnection("Data Source=dota_data.db");
    int count = 0;
    int divineCount = 0;
    try {
        count = await connection.ExecuteScalarAsync<int>("SELECT COUNT(*) FROM Matches");
        divineCount = await connection.ExecuteScalarAsync<int>("SELECT COUNT(*) FROM Matches WHERE avg_rank_tier >= 70");
    } catch {}
    
    return Results.Json(new { 
        totalMatches = count, 
        highMmrMatches = divineCount,
        isRunning = !FetcherService.IsPaused,
        logs = FetcherService.Logs.ToArray(),
        training = engine.GetState(),
        metaDebug = engine.MetaInfo 
    });
});

app.MapGet("/api/heroes", (ModelEngine engine) => engine.GetHeroes());

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
    if (insight.Probability > 0.65) reasonText = "Radiant has a dominant strategic advantage.";
    else if (insight.Probability > 0.53) reasonText = "Radiant slight statistical edge.";
    else if (insight.Probability < 0.35) reasonText = "Dire has a dominant strategic advantage.";
    else if (insight.Probability < 0.47) reasonText = "Dire slight statistical edge.";
    else reasonText = "Dead even. Skill matchup.";

    return Results.Json(new { 
        radiantWinProbability = insight.Probability,
        reason = reasonText,
        factors = insight.Factors
    });
});

app.Run();

// --- DATA MODELS ---

public class ModelInput
{
    [LoadColumn(0)] public float Rank { get; set; }
    
    // Вектор героїв (160 Radiant + 160 Dire = 320)
    [LoadColumn(1, 320)] [VectorType(320)] public float[]? Heroes { get; set; } 

    // Сила Драфту (середній вінрейт героїв)
    [LoadColumn(321)] public float RadiantWinrateScore { get; set; }
    [LoadColumn(322)] public float DireWinrateScore { get; set; }

    [LoadColumn(323)] public float SynergyRadiant { get; set; }
    [LoadColumn(324)] public float SynergyDire { get; set; }
    [LoadColumn(325)] public float CounterScore { get; set; }
    [LoadColumn(326)] public bool Label { get; set; }
    [LoadColumn(327)] public float RadiantMetaScore { get; set; }
    [LoadColumn(328)] public float DireMetaScore { get; set; }
    [LoadColumn(329)] public float RadiantStuns { get; set; }
    [LoadColumn(330)] public float DireStuns { get; set; }
    [LoadColumn(331)] public float RadiantCores { get; set; }
    [LoadColumn(332)] public float DireCores { get; set; }
    [LoadColumn(333)] public float RadiantMelee { get; set; }
    [LoadColumn(334)] public float DireMelee { get; set; }
    [LoadColumn(335)] public float RadiantStr { get; set; }
    [LoadColumn(336)] public float RadiantAgi { get; set; }
    [LoadColumn(337)] public float RadiantInt { get; set; }
    [LoadColumn(338)] public float DireStr { get; set; }
    [LoadColumn(339)] public float DireAgi { get; set; }
    [LoadColumn(340)] public float DireInt { get; set; }
    
    // Вага матчу для балансування
    [LoadColumn(341)] public float Weight { get; set; } = 1f; 
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
    public double LastAuc { get; set; }
    public double LastF1 { get; set; }
}

public class HeroDef
{
    public int id { get; set; }
    public string localized_name { get; set; } = "";
    public string[] roles { get; set; } = Array.Empty<string>();
    public string primary_attr { get; set; } = ""; 
    public string attack_type { get; set; } = ""; 
}

// --- KNOWLEDGE BASE ---
public class KnowledgeBase
{
    public Dictionary<string, float> SynergyMap { get; set; } = new();
    public Dictionary<string, float> CounterMap { get; set; } = new();
    public Dictionary<int, float> HeroWinStats { get; set; } = new();
    public Dictionary<int, HeroDef> Heroes { get; set; } = new();

    public async Task EnsureHeroesLoaded(IHttpClientFactory httpClientFactory)
    {
        if (Heroes.Count > 100) return; 
        try
        {
            using var client = httpClientFactory.CreateClient();
            var json = await client.GetStringAsync("https://api.opendota.com/api/heroes");
            var heroList = JsonSerializer.Deserialize<List<HeroDef>>(json);
            if (heroList != null) Heroes = heroList.ToDictionary(h => h.id, h => h);
        }
        catch {}
    }

    public void Build(List<(int[] r, int[] d, bool win)> matches)
    {
        var pairWins = new ConcurrentDictionary<string, int>();
        var pairTotal = new ConcurrentDictionary<string, int>();
        var counterWins = new ConcurrentDictionary<string, int>();
        var counterTotal = new ConcurrentDictionary<string, int>();
        var heroPicks = new ConcurrentDictionary<int, int>();
        var heroWins = new ConcurrentDictionary<int, int>();

        Parallel.ForEach(matches, match => 
        {
            for (int i = 0; i < match.r.Length; i++)
            {
                int h1 = match.r[i];
                heroPicks.AddOrUpdate(h1, 1, (_, v) => v + 1);
                if (match.win) heroWins.AddOrUpdate(h1, 1, (_, v) => v + 1);
                for (int j = i + 1; j < match.r.Length; j++) UpdatePair(match.r[i], match.r[j], match.win, pairWins, pairTotal);
            }
            for (int i = 0; i < match.d.Length; i++)
            {
                int h1 = match.d[i];
                heroPicks.AddOrUpdate(h1, 1, (_, v) => v + 1);
                if (!match.win) heroWins.AddOrUpdate(h1, 1, (_, v) => v + 1);
                for (int j = i + 1; j < match.d.Length; j++) UpdatePair(match.d[i], match.d[j], !match.win, pairWins, pairTotal);
            }
            foreach (var rHero in match.r)
                foreach (var dHero in match.d)
                {
                    string key = $"{rHero}|{dHero}";
                    counterTotal.AddOrUpdate(key, 1, (k, v) => v + 1);
                    if (match.win) counterWins.AddOrUpdate(key, 1, (k, v) => v + 1);
                }
        });

        SynergyMap.Clear();
        foreach (var kvp in pairTotal) { if (kvp.Value >= 30) SynergyMap[kvp.Key] = ((float)pairWins.GetValueOrDefault(kvp.Key, 0) / kvp.Value) - 0.5f; }
        CounterMap.Clear();
        foreach (var kvp in counterTotal) { if (kvp.Value >= 30) CounterMap[kvp.Key] = ((float)counterWins.GetValueOrDefault(kvp.Key, 0) / kvp.Value) - 0.5f; }
        HeroWinStats.Clear();
        foreach (var kvp in heroPicks) { if (kvp.Value >= 50) HeroWinStats[kvp.Key] = ((float)heroWins.GetValueOrDefault(kvp.Key, 0) / kvp.Value) - 0.5f; }
    }

    private void UpdatePair(int h1, int h2, bool win, ConcurrentDictionary<string, int> wins, ConcurrentDictionary<string, int> total)
    {
        string key = $"{Math.Min(h1, h2)}|{Math.Max(h1, h2)}";
        total.AddOrUpdate(key, 1, (k, v) => v + 1);
        if (win) wins.AddOrUpdate(key, 1, (k, v) => v + 1);
    }

    public float CalculateTeamSynergy(int[] team) {
        float score = 0;
        for (int i = 0; i < team.Length; i++)
            for (int j = i + 1; j < team.Length; j++)
                if (SynergyMap.TryGetValue($"{Math.Min(team[i], team[j])}|{Math.Max(team[i], team[j])}", out float s)) score += s;
        return score;
    }

    public float CalculateCounterScore(int[] rad, int[] dire) {
        float score = 0;
        foreach (var r in rad) foreach (var d in dire) if (CounterMap.TryGetValue($"{r}|{d}", out float s)) score += s;
        return score;
    }

    public float CalculateMetaScore(int[] team) {
        float score = 0;
        foreach (var hero in team) if (HeroWinStats.TryGetValue(hero, out float s)) score += s;
        return score;
    }
    
    public float CalculateTeamWinrateScore(int[] team) {
        float score = 0;
        int count = 0;
        foreach (var hero in team) {
            if (HeroWinStats.TryGetValue(hero, out float s)) {
                score += s;
                count++;
            }
        }
        return count > 0 ? score : 0;
    }

    public (float stuns, float cores) GetComposition(int[] team) {
        float stuns = 0, cores = 0;
        foreach (var id in team) {
            if (Heroes.TryGetValue(id, out var h)) {
                if (h.roles.Contains("Disabler")) stuns += 1.0f;
                if (h.roles.Contains("Carry")) cores += 1.0f;
            }
        }
        return (stuns, cores);
    }

    public (float melee, float str, float agi, float intel) GetAdvStats(int[] team) {
        float melee = 0, str = 0, agi = 0, intel = 0;
        foreach (var id in team) {
            if (Heroes.TryGetValue(id, out var h)) {
                if (h.attack_type == "Melee") melee += 1.0f;
                if (h.primary_attr == "str") str += 1.0f;
                if (h.primary_attr == "agi") agi += 1.0f;
                if (h.primary_attr == "int") intel += 1.0f;
            }
        }
        return (melee, str, agi, intel);
    }
}

// --- ML ENGINE ---

public class ModelEngine
{
    private readonly string _modelPath = "dota_model.zip";
    private readonly string _kbPath = "knowledge_base.json";
    
    private readonly ILogger<ModelEngine> _logger;
    private readonly IHttpClientFactory _httpClientFactory;
    
    private MLContext _mlContext;
    private ITransformer? _trainedModel;
    private PredictionEngine<ModelInput, ModelOutput>? _predEngine;
    private TrainingState _state = new();
    private KnowledgeBase _kb = new();
    public string MetaInfo { get; private set; } = "Not Loaded"; 
    private const int TOTAL_HERO_ID_SLOTS = 160;

    public ModelEngine(ILogger<ModelEngine> logger, IHttpClientFactory httpClientFactory)
    {
        _logger = logger;
        _httpClientFactory = httpClientFactory;
        _mlContext = new MLContext(seed: 42);
        LoadModel();
    }

    public TrainingState GetState() => _state;
    public Dictionary<int, HeroDef> GetHeroes() => _kb.Heroes;

    private void LogInfo(string message)
    {
        _logger.LogInformation(message);
        var time = DateTime.Now.ToString("HH:mm:ss");
        FetcherService.Logs.Enqueue($"[{time}] [AI] {message}");
        if (FetcherService.Logs.Count > 100) FetcherService.Logs.TryDequeue(out _);
        _state.Status = message;
    }

    private void LoadModel()
    {
        string metaPath = _modelPath + ".meta";
        if (File.Exists(metaPath))
        {
            try {
                var metaText = File.ReadAllText(metaPath);
                MetaInfo = metaText; 
                
                var meta = metaText.Split('|');
                if (meta.Length >= 1 && int.TryParse(meta[0], NumberStyles.Any, CultureInfo.InvariantCulture, out int count))
                    _state.LastTrainedOnCount = count;
                
                if (meta.Length >= 2 && double.TryParse(meta[1], NumberStyles.Any, CultureInfo.InvariantCulture, out double acc))
                    _state.LastAccuracy = acc;

                if (meta.Length >= 3 && double.TryParse(meta[2], NumberStyles.Any, CultureInfo.InvariantCulture, out double auc))
                    _state.LastAuc = auc;
                    
                if (meta.Length >= 4 && double.TryParse(meta[3], NumberStyles.Any, CultureInfo.InvariantCulture, out double f1))
                    _state.LastF1 = f1;
                
                LogInfo($"Meta loaded: Acc={_state.LastAccuracy:P1}, AUC={_state.LastAuc:F3}");
            } catch (Exception ex) {
                MetaInfo = $"Error reading meta: {ex.Message}";
                LogInfo("Meta file corruption detected.");
            }
        }
        else 
        {
            MetaInfo = "Meta file not found";
        }

        if (File.Exists(_modelPath)) 
        {
            try 
            {
                _trainedModel = _mlContext.Model.Load(_modelPath, out var schema);
                _predEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_trainedModel);
                LogInfo("Model binary loaded successfully.");
            }
            catch (Exception ex) { 
                _logger.LogError(ex, "Failed to load model binary"); 
                if (_state.LastTrainedOnCount == 0) _state.Status = "Load Error";
            }
        }

        if (File.Exists(_kbPath))
        {
            try {
                var json = File.ReadAllText(_kbPath);
                _kb = JsonSerializer.Deserialize<KnowledgeBase>(json) ?? new KnowledgeBase();
            } catch {}
        }
    }

    public PredictionResult PredictWithInsights(PredictionRequest req)
    {
        if (_predEngine == null) return new PredictionResult { Probability = 0.5f };

        var baseInput = CreateInput(req);
        var basePrediction = _predEngine.Predict(baseInput);
        float baseProb = basePrediction.Probability;

        var result = new PredictionResult { Probability = baseProb };

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
        var (rStun, rCore) = _kb.GetComposition(req.Radiant);
        var (dStun, dCore) = _kb.GetComposition(req.Dire);
        var (rMelee, rStr, rAgi, rInt) = _kb.GetAdvStats(req.Radiant);
        var (dMelee, dStr, dAgi, dInt) = _kb.GetAdvStats(req.Dire);

        var input = new ModelInput
        {
            Rank = req.Rank * 10, 
            Heroes = new float[TOTAL_HERO_ID_SLOTS * 2], 
            RadiantWinrateScore = _kb.CalculateTeamWinrateScore(req.Radiant),
            DireWinrateScore = _kb.CalculateTeamWinrateScore(req.Dire),
            SynergyRadiant = _kb.CalculateTeamSynergy(req.Radiant),
            SynergyDire = _kb.CalculateTeamSynergy(req.Dire),
            CounterScore = _kb.CalculateCounterScore(req.Radiant, req.Dire),
            RadiantMetaScore = _kb.CalculateMetaScore(req.Radiant),
            DireMetaScore = _kb.CalculateMetaScore(req.Dire),
            RadiantStuns = rStun, DireStuns = dStun,
            RadiantCores = rCore, DireCores = dCore,
            RadiantMelee = rMelee, DireMelee = dMelee,
            RadiantStr = rStr, RadiantAgi = rAgi, RadiantInt = rInt,
            DireStr = dStr, DireAgi = dAgi, DireInt = dInt,
            Weight = 1f 
        };

        foreach (var id in req.Radiant) if (id > 0 && id < TOTAL_HERO_ID_SLOTS) input.Heroes![id] = 1;
        foreach (var id in req.Dire)    if (id > 0 && id < TOTAL_HERO_ID_SLOTS) input.Heroes![TOTAL_HERO_ID_SLOTS + id] = 1;

        return input;
    }

    public async void Train()
    {
        try
        {
            _state.IsTraining = true;
            _state.ProgressPercent = 0;
            LogInfo("Starting training sequence (Deep Small Trees + Weighting)...");
            
            await _kb.EnsureHeroesLoaded(_httpClientFactory);

            LogInfo("Fetching matches (High Rank Only: Divine+)...");
            using var connection = new SqliteConnection("Data Source=dota_data.db");
            
            var sql = @"
                SELECT avg_rank_tier, radiant_team, dire_team, radiant_win FROM Matches 
                WHERE avg_rank_tier >= 70 AND duration > 1200";
                
            var rawData = connection.Query<(int rank, string rTeam, string dTeam, bool win)>(sql).ToList();
            
            if (rawData.Count < 50)
            {
                LogInfo("Not enough data. Training aborted.");
                _state.IsTraining = false;
                return;
            }

            double radiantWins = rawData.Count(x => x.win);
            double direWins = rawData.Count(x => !x.win);
            LogInfo($"Win rate: Radiant {radiantWins/(double)rawData.Count:P2}");
            
            float direWeightMultiplier = (float)(radiantWins / Math.Max(direWins, 1));
            LogInfo($"Applying Weight Correction: Dire wins x{direWeightMultiplier:F2}");

            _state.ProgressPercent = 10;
            
            var parsedMatches = new List<(int[] r, int[] d, bool win)>();
            foreach(var m in rawData) {
                var r = JsonSerializer.Deserialize<int[]>(m.rTeam);
                var d = JsonSerializer.Deserialize<int[]>(m.dTeam);
                if(r != null && d != null) parsedMatches.Add((r, d, m.win));
            }

            _kb.Build(parsedMatches);
            File.WriteAllText(_kbPath, JsonSerializer.Serialize(_kb));

            _state.ProgressPercent = 20;
            LogInfo("Vectorizing features with Weights...");

            var trainingData = new List<ModelInput>();
            int idx = 0;
            foreach (var match in rawData)
            {
                var pm = parsedMatches[idx++];
                var (rStun, rCore) = _kb.GetComposition(pm.r);
                var (dStun, dCore) = _kb.GetComposition(pm.d);
                var (rMelee, rStr, rAgi, rInt) = _kb.GetAdvStats(pm.r);
                var (dMelee, dStr, dAgi, dInt) = _kb.GetAdvStats(pm.d);

                var input = new ModelInput 
                { 
                    Rank = match.rank, Label = match.win, 
                    Heroes = new float[TOTAL_HERO_ID_SLOTS * 2], 
                    RadiantWinrateScore = _kb.CalculateTeamWinrateScore(pm.r),
                    DireWinrateScore = _kb.CalculateTeamWinrateScore(pm.d),
                    SynergyRadiant = _kb.CalculateTeamSynergy(pm.r), SynergyDire = _kb.CalculateTeamSynergy(pm.d),
                    CounterScore = _kb.CalculateCounterScore(pm.r, pm.d),
                    RadiantMetaScore = _kb.CalculateMetaScore(pm.r), DireMetaScore = _kb.CalculateMetaScore(pm.d),
                    RadiantStuns = rStun, DireStuns = dStun, RadiantCores = rCore, DireCores = dCore,
                    RadiantMelee = rMelee, DireMelee = dMelee, RadiantStr = rStr, RadiantAgi = rAgi, RadiantInt = rInt, DireStr = dStr, DireAgi = dAgi, DireInt = dInt,
                    
                    // [BALANCED] Зважування ON
                    Weight = match.win ? 1.0f : direWeightMultiplier
                };

                foreach (var id in pm.r) if (id > 0 && id < TOTAL_HERO_ID_SLOTS) input.Heroes![id] = 1;
                foreach (var id in pm.d) if (id > 0 && id < TOTAL_HERO_ID_SLOTS) input.Heroes![TOTAL_HERO_ID_SLOTS + id] = 1;
                
                trainingData.Add(input);
            }

            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            var split = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.15);

            var options = new LightGbmBinaryTrainer.Options
            {
                LabelColumnName = "Label", 
                FeatureColumnName = "Features",
                ExampleWeightColumnName = "Weight",
                
                // [DEEP SMALL TREES] Стратегія "багато маленьких кроків"
                NumberOfLeaves = 40, // Маленькі дерева = менше перенавчання
                MinimumExampleCountPerLeaf = 50, // Тільки надійні дані
                LearningRate = 0.01, // Дуже повільно
                NumberOfIterations = 8000, // Дуже довго (компенсація малого кроку)
                Booster = new GradientBooster.Options { L2Regularization = 0.5f, FeatureFraction = 0.85 }
            };

            // [NEW] Повернули калібратор для кращого AUC
            var pipeline = _mlContext.Transforms.Concatenate("Features", 
                    "Rank", "Heroes", "RadiantWinrateScore", "DireWinrateScore", "SynergyRadiant", "SynergyDire", "CounterScore",
                    "RadiantMetaScore", "DireMetaScore", "RadiantStuns", "DireStuns", "RadiantCores", "DireCores",
                    "RadiantMelee", "DireMelee", "RadiantStr", "RadiantAgi", "RadiantInt", "DireStr", "DireAgi", "DireInt")
                .Append(_mlContext.BinaryClassification.Trainers.LightGbm(options))
                .Append(_mlContext.BinaryClassification.Calibrators.Platt()); 

            _state.ProgressPercent = 50;
            LogInfo("Training Deep Small Trees Model...");

            _trainedModel = pipeline.Fit(split.TrainSet);

            _state.ProgressPercent = 80;
            LogInfo("Evaluating on 15% test set...");

            var predictions = _trainedModel.Transform(split.TestSet);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions, "Label");

            _state.ProgressPercent = 90;
            LogInfo($"Metrics (Real Test): Acc={metrics.Accuracy:P2}, AUC={metrics.AreaUnderRocCurve:F3}");
            LogInfo("Saving...");

            _mlContext.Model.Save(_trainedModel, dataView.Schema, _modelPath);
            File.WriteAllText(_modelPath + ".meta", 
                $"{rawData.Count}|{metrics.Accuracy.ToString(CultureInfo.InvariantCulture)}|{metrics.AreaUnderRocCurve.ToString(CultureInfo.InvariantCulture)}|{metrics.F1Score.ToString(CultureInfo.InvariantCulture)}");
            
            _predEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_trainedModel);

            _state.LastTrainedOnCount = rawData.Count;
            _state.LastAccuracy = metrics.Accuracy;
            _state.LastAuc = metrics.AreaUnderRocCurve;
            _state.LastF1 = metrics.F1Score;

            LogInfo("Training Complete.");
            _state.Status = $"Ready (AUC: {metrics.AreaUnderRocCurve:F3})";
            _state.ProgressPercent = 100;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Training crash");
            LogInfo("CRITICAL ERROR. Check console.");
            _state.Status = "Training Failed";
        }
        finally
        {
            _state.IsTraining = false;
        }
    }
}