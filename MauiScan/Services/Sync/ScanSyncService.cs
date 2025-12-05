using Microsoft.AspNetCore.SignalR.Client;
using MauiScan.Models;
using System.Net.Http.Headers;
using System.Net.Http.Json;

namespace MauiScan.Services.Sync;

public class ScanSyncService
{
    private readonly string _serverUrl;
    private readonly HttpClient _httpClient;
    private HubConnection? _hubConnection;

    public event Action<ScanImageDto>? NewScanReceived;
    public event Action<bool>? ConnectionStateChanged;

    public bool IsConnected => _hubConnection?.State == HubConnectionState.Connected;

    public ScanSyncService(string serverUrl)
    {
        _serverUrl = serverUrl.TrimEnd('/');
        _httpClient = new HttpClient
        {
            BaseAddress = new Uri(_serverUrl),
            Timeout = TimeSpan.FromMinutes(2)
        };
    }

    public async Task ConnectAsync()
    {
        if (_hubConnection != null)
        {
            await _hubConnection.StopAsync();
            await _hubConnection.DisposeAsync();
        }

        _hubConnection = new HubConnectionBuilder()
            .WithUrl($"{_serverUrl}/hubs/scan")
            .WithAutomaticReconnect(new[] {
                TimeSpan.Zero,
                TimeSpan.FromSeconds(2),
                TimeSpan.FromSeconds(5),
                TimeSpan.FromSeconds(10)
            })
            .Build();

        // 监听新扫描事件
        _hubConnection.On<ScanImageDto>("ReceiveNewScan", scanImage =>
        {
            NewScanReceived?.Invoke(scanImage);
        });

        // 监听连接状态变化
        _hubConnection.Reconnecting += _ =>
        {
            ConnectionStateChanged?.Invoke(false);
            return Task.CompletedTask;
        };

        _hubConnection.Reconnected += _ =>
        {
            ConnectionStateChanged?.Invoke(true);
            return Task.CompletedTask;
        };

        _hubConnection.Closed += _ =>
        {
            ConnectionStateChanged?.Invoke(false);
            return Task.CompletedTask;
        };

        try
        {
            await _hubConnection.StartAsync();
            ConnectionStateChanged?.Invoke(true);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"SignalR 连接失败: {ex.Message}");
            ConnectionStateChanged?.Invoke(false);
        }
    }

    public async Task<bool> UploadScanAsync(byte[] imageData, int width, int height)
    {
        try
        {
            using var content = new MultipartFormDataContent();

            // 添加图片文件
            var imageContent = new ByteArrayContent(imageData);
            imageContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg");
            content.Add(imageContent, "file", $"scan_{DateTime.Now:yyyyMMdd_HHmmss}.jpg");

            // 添加宽高参数
            content.Add(new StringContent(width.ToString()), "width");
            content.Add(new StringContent(height.ToString()), "height");

            var response = await _httpClient.PostAsync("/api/scans/upload", content);

            if (response.IsSuccessStatusCode)
            {
                System.Diagnostics.Debug.WriteLine("图片上传成功");
                return true;
            }
            else
            {
                var error = await response.Content.ReadAsStringAsync();
                System.Diagnostics.Debug.WriteLine($"图片上传失败: {response.StatusCode}, {error}");
                return false;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"图片上传异常: {ex.Message}");
            return false;
        }
    }

    public async Task<List<ScanImageDto>> GetRecentScansAsync(int limit = 10)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/api/scans/recent?limit={limit}");

            if (response.IsSuccessStatusCode)
            {
                var scans = await response.Content.ReadFromJsonAsync<List<ScanImageDto>>();
                return scans ?? new List<ScanImageDto>();
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"获取最近扫描列表失败: {ex.Message}");
        }

        return new List<ScanImageDto>();
    }

    public async Task<byte[]?> DownloadScanAsync(string downloadUrl)
    {
        try
        {
            var response = await _httpClient.GetAsync(downloadUrl);

            if (response.IsSuccessStatusCode)
            {
                return await response.Content.ReadAsByteArrayAsync();
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"下载图片失败: {ex.Message}");
        }

        return null;
    }

    public async Task DisconnectAsync()
    {
        if (_hubConnection != null)
        {
            await _hubConnection.StopAsync();
            await _hubConnection.DisposeAsync();
            _hubConnection = null;
        }
    }
}
