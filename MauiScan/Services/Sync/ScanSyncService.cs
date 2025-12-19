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
        // 如果已经连接，直接返回
        if (_hubConnection?.State == HubConnectionState.Connected)
        {
            return;
        }

        // 如果连接存在但未连接，尝试重新连接
        if (_hubConnection != null)
        {
            try
            {
                if (_hubConnection.State == HubConnectionState.Disconnected)
                {
                    await _hubConnection.StartAsync();
                    ConnectionStateChanged?.Invoke(true);
                    return;
                }
            }
            catch
            {
                // 重连失败，重新创建连接
                await _hubConnection.StopAsync();
                await _hubConnection.DisposeAsync();
                _hubConnection = null;
            }
        }

        // 创建新连接
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
            System.Diagnostics.Debug.WriteLine($"[SignalR] 收到 ReceiveNewScan 事件: {scanImage.FileName}");
            NewScanReceived?.Invoke(scanImage);
        });

        // 监听连接状态变化
        _hubConnection.Reconnecting += _ =>
        {
            System.Diagnostics.Debug.WriteLine("[SignalR] 正在重新连接...");
            ConnectionStateChanged?.Invoke(false);
            return Task.CompletedTask;
        };

        _hubConnection.Reconnected += _ =>
        {
            System.Diagnostics.Debug.WriteLine("[SignalR] 已重新连接");
            ConnectionStateChanged?.Invoke(true);
            return Task.CompletedTask;
        };

        _hubConnection.Closed += _ =>
        {
            System.Diagnostics.Debug.WriteLine("[SignalR] 连接已关闭");
            ConnectionStateChanged?.Invoke(false);
            return Task.CompletedTask;
        };

        try
        {
            await _hubConnection.StartAsync();
            System.Diagnostics.Debug.WriteLine($"[SignalR] 连接成功: {_serverUrl}/hubs/scan");
            ConnectionStateChanged?.Invoke(true);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[SignalR] 连接失败: {ex.Message}");
            ConnectionStateChanged?.Invoke(false);
        }
    }

    /// <summary>
    /// 上传扫描图片，返回上传成功后的文件名（用于过滤自己上传的通知）
    /// </summary>
    public async Task<string?> UploadScanAsync(byte[] originalImageData, byte[] processedImageData, int width, int height)
    {
        try
        {
            using var content = new MultipartFormDataContent();

            System.Diagnostics.Debug.WriteLine($"准备上传 - Width: {width}, Height: {height}, OriginalSize: {originalImageData.Length}, ProcessedSize: {processedImageData.Length}");

            // 添加原图
            var originalContent = new ByteArrayContent(originalImageData);
            originalContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg");
            content.Add(originalContent, "originalImage", $"scan_{DateTime.Now:yyyyMMdd_HHmmss}_original.jpg");

            // 添加处理后的图片
            var processedContent = new ByteArrayContent(processedImageData);
            processedContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg");
            content.Add(processedContent, "processedImage", $"scan_{DateTime.Now:yyyyMMdd_HHmmss}_processed.jpg");

            // 添加宽高参数
            content.Add(new StringContent(width.ToString()), "width");
            content.Add(new StringContent(height.ToString()), "height");

            var response = await _httpClient.PostAsync("/api/scans/upload", content);

            if (response.IsSuccessStatusCode)
            {
                // 解析响应获取文件名
                var uploadResponse = await response.Content.ReadFromJsonAsync<UploadResponse>();
                var fileName = uploadResponse?.ScanImage?.FileName;
                System.Diagnostics.Debug.WriteLine($"图片上传成功: {fileName}");
                return fileName;
            }
            else
            {
                var error = await response.Content.ReadAsStringAsync();
                System.Diagnostics.Debug.WriteLine($"图片上传失败: {response.StatusCode}, {error}");
                return null;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"图片上传异常: {ex.Message}");
            return null;
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

    public async Task<bool> DeleteScanAsync(string fileName)
    {
        try
        {
            var response = await _httpClient.DeleteAsync($"/api/scans/{fileName}");
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"删除扫描失败: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// 上传训练数据（手动标注的原图 + 角点坐标）
    /// </summary>
    public async Task<bool> UploadTrainingDataAsync(byte[] originalImage, float[] corners, byte[] processedImage)
    {
        try
        {
            using var content = new MultipartFormDataContent();

            // 添加原图
            var originalImageContent = new ByteArrayContent(originalImage);
            originalImageContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg");
            content.Add(originalImageContent, "originalImage", $"training_{DateTime.Now:yyyyMMdd_HHmmss}_original.jpg");

            // 添加处理后的图片（用于验证）
            var processedImageContent = new ByteArrayContent(processedImage);
            processedImageContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg");
            content.Add(processedImageContent, "processedImage", $"training_{DateTime.Now:yyyyMMdd_HHmmss}_processed.jpg");

            // 添加角点坐标
            content.Add(new StringContent(string.Join(",", corners)), "corners");

            // 添加设备信息
            content.Add(new StringContent(DeviceInfo.Current.Model), "deviceModel");
            content.Add(new StringContent(DeviceInfo.Current.Platform.ToString()), "devicePlatform");

            var response = await _httpClient.PostAsync("/api/training/upload", content);

            if (response.IsSuccessStatusCode)
            {
                System.Diagnostics.Debug.WriteLine("训练数据上传成功");
                return true;
            }
            else
            {
                var error = await response.Content.ReadAsStringAsync();
                System.Diagnostics.Debug.WriteLine($"训练数据上传失败: {response.StatusCode}, {error}");
                return false;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"训练数据上传异常: {ex.Message}");
            return false;
        }
    }
}
