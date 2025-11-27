using Dapper;
using Microsoft.Data.Sqlite;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections.Concurrent;
using System.Web;

namespace DotaPredictor.Services;

public class FetcherService : BackgroundService
{
    private readonly ILogger<FetcherService> _logger;
    private readonly IHttpClientFactory _httpClientFactory;
    private const string ConnectionString = "Data Source=dota_data.db";
    
    // UI CONTROL
    public static bool IsPaused { get; set; } = true; 
    public static ConcurrentQueue<string> Logs { get; } = new();

    // Зберігаємо ID останнього завантаженого матчу, щоб рухатись в минуле
    private long _lastMatchId = 0; 

    // Налаштування безпеки API
    private const int RowsPerRequest = 2000; // Беремо максимум за раз
    private const int DelayBetweenCallsMs = 4000; // 4 секунди паузи (дуже безпечно для Rate Limit)

    public FetcherService(ILogger<FetcherService> logger, IHttpClientFactory httpClientFactory)
    {
        _logger = logger;
        _httpClientFactory = httpClientFactory;
    }

    private void AddLog(string message)
    {
        var logMsg = $"[{DateTime.Now:HH:mm:ss}] {message}";
        Logs.Enqueue(logMsg);
        if (Logs.Count > 50) Logs.TryDequeue(out _);
        _logger.LogInformation(message);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        AddLog("🚀 High-Speed Fetcher initialized.");
        AddLog($"Strategy: {RowsPerRequest} matches per call, {DelayBetweenCallsMs}ms delay.");

        // При старті знаходимо найменший ID в базі, щоб продовжити скачування історії
        await InitLastMatchId();

        while (!stoppingToken.IsCancellationRequested)
        {
            if (IsPaused)
            {
                await Task.Delay(1000, stoppingToken);
                continue;
            }

            try
            {
                await FetchMatchesBulk();
            }
            catch (Exception ex)
            {
                AddLog($"ERROR: {ex.Message}");
                await Task.Delay(10000, stoppingToken); // При помилці чекаємо 10 сек
            }

            // Робимо паузу, щоб не перевищити ліміт 60 запитів/хв
            await Task.Delay(DelayBetweenCallsMs, stoppingToken);
        }
    }

    private async Task InitLastMatchId()
    {
        using var connection = new SqliteConnection(ConnectionString);
        // Шукаємо найменший ID, щоб качати старіші матчі
        var minId = await connection.ExecuteScalarAsync<long?>("SELECT MIN(match_id) FROM Matches");
        
        if (minId.HasValue && minId.Value > 0)
        {
            _lastMatchId = minId.Value;
            AddLog($"Resuming download from Match ID: {_lastMatchId}");
        }
    }

    private async Task FetchMatchesBulk()
    {
        using var client = _httpClientFactory.CreateClient();
        client.Timeout = TimeSpan.FromSeconds(60); // SQL запити можуть бути довгими

        // Якщо це перший запуск і _lastMatchId = 0, починаємо з "зараз"
        string whereClause = _lastMatchId > 0 
            ? $"match_id < {_lastMatchId}" 
            : "TRUE"; 

        // SQL для OpenDota Explorer
        // Беремо тільки потрібні колонки, щоб зменшити навантаження на API
        var openDotaSql = $@"
            SELECT 
                match_id, 
                radiant_win, 
                start_time, 
                duration, 
                avg_rank_tier, 
                radiant_team, 
                dire_team 
            FROM public_matches 
            WHERE lobby_type = 7 
            AND game_mode = 22 
            AND duration > 900 
            AND {whereClause}
            ORDER BY match_id DESC 
            LIMIT {RowsPerRequest}";

        // Кодуємо SQL для URL
        // Важливо: OpenDota іноді вимагає, щоб пробіли були %20, але UrlEncode це робить
        var encodedSql = HttpUtility.UrlEncode(openDotaSql);
        var url = $"https://api.opendota.com/api/explorer?sql={encodedSql}";

        AddLog($"Requesting {RowsPerRequest} matches...");

        // Використовуємо User-Agent, щоб нас не блокували як бота
        var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.Add("User-Agent", "Dota2Predictor/1.0 (Student Project)");

        var response = await client.SendAsync(request);

        if (!response.IsSuccessStatusCode)
        {
            AddLog($"API Warning: {response.StatusCode}. Cooling down for 10s...");
            await Task.Delay(10000);
            return;
        }

        var content = await response.Content.ReadAsStringAsync();
        
        // Обробка помилок JSON
        JObject json;
        try 
        {
            json = JObject.Parse(content);
        }
        catch
        {
            AddLog("Error parsing JSON response. Retrying...");
            return;
        }

        var rows = json["rows"] as JArray;

        if (rows == null || rows.Count == 0)
        {
            AddLog("No more matches found via Explorer. Stopping.");
            IsPaused = true; 
            return;
        }

        using var connection = new SqliteConnection(ConnectionString);
        await connection.OpenAsync();
        
        // Використовуємо транзакцію для супер-швидкої вставки
        using var transaction = connection.BeginTransaction();
        
        int count = 0;
        long minIdInBatch = long.MaxValue;

        foreach (var row in rows)
        {
            long matchId = row["match_id"]?.Value<long>() ?? 0;
            if (matchId == 0) continue;

            if (matchId < minIdInBatch) minIdInBatch = matchId;

            var sql = @"
                INSERT OR IGNORE INTO Matches 
                (match_id, radiant_win, start_time, duration, avg_rank_tier, radiant_team, dire_team, raw_json)
                VALUES 
                (@Id, @Win, @Start, @Dur, @Rank, @RadTeam, @DireTeam, @Raw)";

            // Нормалізація рядків команд (іноді API віддає "1,2,3" замість "[1,2,3]")
            string rTeam = row["radiant_team"]?.ToString() ?? "";
            string dTeam = row["dire_team"]?.ToString() ?? "";
            
            if (!string.IsNullOrEmpty(rTeam) && !rTeam.StartsWith("[")) rTeam = $"[{rTeam}]";
            if (!string.IsNullOrEmpty(dTeam) && !dTeam.StartsWith("[")) dTeam = $"[{dTeam}]";

            await connection.ExecuteAsync(sql, new {
                Id = matchId,
                Win = row["radiant_win"]?.Value<bool>() ?? false,
                Start = row["start_time"]?.Value<int>() ?? 0,
                Dur = row["duration"]?.Value<int>() ?? 0,
                Rank = row["avg_rank_tier"]?.Value<int>() ?? 0,
                RadTeam = rTeam,
                DireTeam = dTeam,
                Raw = row.ToString(Formatting.None)
            }, transaction);

            count++;
        }

        transaction.Commit();
        
        if (minIdInBatch < long.MaxValue && minIdInBatch > 0)
        {
            _lastMatchId = minIdInBatch;
        }

        AddLog($"✅ Saved batch of {count} matches. (Last ID: {_lastMatchId})");
    }
}