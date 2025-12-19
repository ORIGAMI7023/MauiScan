using MauiScan.Server.Models;
using SixLabors.ImageSharp;

namespace MauiScan.Server.Services;

public interface IFileStorageService
{
    Task<ScanImageDto> SaveFileAsync(IFormFile file, int width, int height);
    Task<(byte[] fileData, string contentType)?> GetFileAsync(string fileName);
    Task<List<ScanImageDto>> GetRecentScansAsync(int limit = 10);
    Task<bool> DeleteFileAsync(string fileName);
    Task SaveTrainingImageAsync(IFormFile file, string referenceName);
}

public class FileStorageService : IFileStorageService
{
    private readonly string _storageDirectory;
    private readonly string _trainingDirectory;
    private readonly long _maxFileSizeBytes;

    public FileStorageService(IConfiguration configuration)
    {
        _storageDirectory = configuration["Storage:ScansDirectory"] ?? "data/scans";
        _trainingDirectory = configuration["Storage:TrainingDataDirectory"] ?? "data/training";
        _maxFileSizeBytes = (configuration.GetValue<int?>("Storage:MaxFileSizeMB") ?? 20) * 1024 * 1024;

        // 确保存储目录存在
        if (!Directory.Exists(_storageDirectory))
        {
            Directory.CreateDirectory(_storageDirectory);
        }

        // 确保训练目录存在
        if (!Directory.Exists(_trainingDirectory))
        {
            Directory.CreateDirectory(_trainingDirectory);
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

        var scanImage = new ScanImageDto
        {
            FileName = fileName,
            FileSize = fileInfo.Length,
            Width = width,
            Height = height,
            ScannedAt = DateTime.UtcNow.AddHours(8),  // 转换为北京时间 (UTC+8)
            DownloadUrl = $"/api/scans/{fileName}"
        };

        // 保存元数据到 JSON 文件
        await SaveMetadataAsync(fileName, scanImage);

        return scanImage;
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
            // 尝试加载元数据
            var metadata = await LoadMetadataAsync(file.Name);

            // 如果元数据不存在，尝试从图像文件重建
            if (metadata == null)
            {
                metadata = await ReconstructMetadataFromImageAsync(file.FullName);
            }

            result.Add(metadata ?? new ScanImageDto
            {
                FileName = file.Name,
                FileSize = file.Length,
                Width = -1,
                Height = -1,
                ScannedAt = file.LastWriteTimeUtc.AddHours(8),
                DownloadUrl = $"/api/scans/{file.Name}"
            });
        }

        return result;
    }

    private async Task SaveMetadataAsync(string fileName, ScanImageDto scanImage)
    {
        var metadataPath = Path.Combine(_storageDirectory, $"{fileName}.json");
        var json = System.Text.Json.JsonSerializer.Serialize(scanImage, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });
        await File.WriteAllTextAsync(metadataPath, json);
    }

    private async Task<ScanImageDto?> LoadMetadataAsync(string fileName)
    {
        var metadataPath = Path.Combine(_storageDirectory, $"{fileName}.json");

        if (!File.Exists(metadataPath))
        {
            return null;
        }

        try
        {
            var json = await File.ReadAllTextAsync(metadataPath);
            return System.Text.Json.JsonSerializer.Deserialize<ScanImageDto>(json);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// 从图像文件重建元数据（用于没有JSON元数据的旧文件）
    /// </summary>
    private async Task<ScanImageDto?> ReconstructMetadataFromImageAsync(string imagePath)
    {
        try
        {
            var fileInfo = new FileInfo(imagePath);

            // 使用 SixLabors.ImageSharp 读取图像尺寸
            using var stream = File.OpenRead(imagePath);
            var imageInfo = await Image.IdentifyAsync(stream);

            if (imageInfo == null)
            {
                return null;
            }

            var metadata = new ScanImageDto
            {
                FileName = Path.GetFileName(imagePath),
                FileSize = fileInfo.Length,
                Width = imageInfo.Width,
                Height = imageInfo.Height,
                ScannedAt = fileInfo.LastWriteTimeUtc,  // 将在调用方转换为北京时间
                DownloadUrl = $"/api/scans/{Path.GetFileName(imagePath)}"
            };

            // 保存元数据，避免下次重复读取
            await SaveMetadataAsync(Path.GetFileName(imagePath), metadata);

            return metadata;
        }
        catch (Exception ex)
        {
            // 记录错误但返回null，允许使用fallback值
            Console.WriteLine($"Failed to reconstruct metadata for {imagePath}: {ex.Message}");
            return null;
        }
    }

    public async Task<bool> DeleteFileAsync(string fileName)
    {
        // 防止路径穿越攻击
        if (fileName.Contains("..") || fileName.Contains("/") || fileName.Contains("\\"))
        {
            return false;
        }

        var filePath = Path.Combine(_storageDirectory, fileName);
        var metadataPath = Path.Combine(_storageDirectory, $"{fileName}.json");

        bool deleted = false;

        if (File.Exists(filePath))
        {
            File.Delete(filePath);
            deleted = true;
        }

        if (File.Exists(metadataPath))
        {
            File.Delete(metadataPath);
        }

        // 删除对应的训练图片
        var trainingFileName = Path.GetFileNameWithoutExtension(fileName) + "_original.jpg";
        var trainingFilePath = Path.Combine(_trainingDirectory, trainingFileName);
        var trainingMetadataPath = Path.Combine(_trainingDirectory, $"{trainingFileName}.json");

        if (File.Exists(trainingFilePath))
        {
            File.Delete(trainingFilePath);
        }

        if (File.Exists(trainingMetadataPath))
        {
            File.Delete(trainingMetadataPath);
        }

        return deleted;
    }

    public async Task SaveTrainingImageAsync(IFormFile file, string referenceName)
    {
        var fileName = Path.GetFileNameWithoutExtension(referenceName) + "_original.jpg";
        var filePath = Path.Combine(_trainingDirectory, fileName);

        using (var stream = new FileStream(filePath, FileMode.Create))
        {
            await file.CopyToAsync(stream);
        }

        // 保存元数据
        var metadata = new
        {
            originalFileName = file.FileName,
            savedFileName = fileName,
            processedImageReference = referenceName,
            uploadedAt = DateTime.UtcNow.AddHours(8),
            fileSize = new FileInfo(filePath).Length
        };

        var metadataPath = Path.Combine(_trainingDirectory, $"{fileName}.json");
        var json = System.Text.Json.JsonSerializer.Serialize(metadata, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });
        await File.WriteAllTextAsync(metadataPath, json);
    }
}
