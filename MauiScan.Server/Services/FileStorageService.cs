using MauiScan.Server.Models;

namespace MauiScan.Server.Services;

public interface IFileStorageService
{
    Task<ScanImageDto> SaveFileAsync(IFormFile file, int width, int height);
    Task<(byte[] fileData, string contentType)?> GetFileAsync(string fileName);
    Task<List<ScanImageDto>> GetRecentScansAsync(int limit = 10);
}

public class FileStorageService : IFileStorageService
{
    private readonly string _storageDirectory;
    private readonly long _maxFileSizeBytes;

    public FileStorageService(IConfiguration configuration)
    {
        _storageDirectory = configuration["Storage:ScansDirectory"] ?? "data/scans";
        _maxFileSizeBytes = (configuration.GetValue<int?>("Storage:MaxFileSizeMB") ?? 20) * 1024 * 1024;

        // 确保存储目录存在
        if (!Directory.Exists(_storageDirectory))
        {
            Directory.CreateDirectory(_storageDirectory);
        }
    }

    public async Task<ScanImageDto> SaveFileAsync(IFormFile file, int width, int height)
    {
        if (file.Length > _maxFileSizeBytes)
        {
            throw new InvalidOperationException($"文件大小超过限制 {_maxFileSizeBytes / 1024 / 1024}MB");
        }

        // 生成唯一文件名：timestamp_guid.jpg
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var guid = Guid.NewGuid().ToString("N")[..8];
        var fileName = $"{timestamp}_{guid}.jpg";
        var filePath = Path.Combine(_storageDirectory, fileName);

        // 保存文件
        using (var stream = new FileStream(filePath, FileMode.Create))
        {
            await file.CopyToAsync(stream);
        }

        var fileInfo = new FileInfo(filePath);

        return new ScanImageDto
        {
            FileName = fileName,
            FileSize = fileInfo.Length,
            Width = width,
            Height = height,
            ScannedAt = DateTime.UtcNow,
            DownloadUrl = $"/api/scans/{fileName}"
        };
    }

    public async Task<(byte[] fileData, string contentType)?> GetFileAsync(string fileName)
    {
        // 防止路径穿越攻击
        if (fileName.Contains("..") || fileName.Contains("/") || fileName.Contains("\\"))
        {
            return null;
        }

        var filePath = Path.Combine(_storageDirectory, fileName);

        if (!File.Exists(filePath))
        {
            return null;
        }

        var fileData = await File.ReadAllBytesAsync(filePath);
        return (fileData, "image/jpeg");
    }

    public async Task<List<ScanImageDto>> GetRecentScansAsync(int limit = 10)
    {
        var directory = new DirectoryInfo(_storageDirectory);

        if (!directory.Exists)
        {
            return new List<ScanImageDto>();
        }

        var files = directory.GetFiles("*.jpg")
            .OrderByDescending(f => f.LastWriteTimeUtc)
            .Take(limit)
            .ToList();

        var result = new List<ScanImageDto>();

        foreach (var file in files)
        {
            // 尝试从文件名解析宽高（如果有元数据文件）
            // 这里简化处理，返回默认值
            result.Add(new ScanImageDto
            {
                FileName = file.Name,
                FileSize = file.Length,
                Width = 0, // 简化版不存储尺寸
                Height = 0,
                ScannedAt = file.LastWriteTimeUtc,
                DownloadUrl = $"/api/scans/{file.Name}"
            });
        }

        return await Task.FromResult(result);
    }
}
