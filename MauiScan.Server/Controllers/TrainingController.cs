using Microsoft.AspNetCore.Mvc;
using System.Text.Json;

namespace MauiScan.Server.Controllers;

[ApiController]
[Route("api/[controller]")]
public class TrainingController : ControllerBase
{
    private readonly string _trainingDataDirectory;
    private readonly ILogger<TrainingController> _logger;

    public TrainingController(IConfiguration configuration, ILogger<TrainingController> logger)
    {
        _trainingDataDirectory = configuration["Storage:TrainingDataDirectory"] ?? "data/training";
        _logger = logger;

        // 确保训练数据目录存在
        if (!Directory.Exists(_trainingDataDirectory))
        {
            Directory.CreateDirectory(_trainingDataDirectory);
        }
    }

    /// <summary>
    /// 上传训练数据（手动标注的原图 + 角点坐标 + 处理后的图片）
    /// </summary>
    [HttpPost("upload")]
    [RequestSizeLimit(40 * 1024 * 1024)] // 40MB (原图 + 处理后的图片)
    public async Task<ActionResult> Upload(
        [FromForm] IFormFile originalImage,
        [FromForm] IFormFile processedImage,
        [FromForm] string corners,
        [FromForm] string? deviceModel = null,
        [FromForm] string? devicePlatform = null)
    {
        try
        {
            if (originalImage == null || originalImage.Length == 0)
            {
                return BadRequest(new { success = false, message = "原图为空" });
            }

            if (processedImage == null || processedImage.Length == 0)
            {
                return BadRequest(new { success = false, message = "处理后的图片为空" });
            }

            if (string.IsNullOrWhiteSpace(corners))
            {
                return BadRequest(new { success = false, message = "角点坐标为空" });
            }

            // 生成唯一标识符
            var timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            var guid = Guid.NewGuid().ToString("N")[..8];
            var baseName = $"{timestamp}_{guid}";

            // 保存原图
            var originalImagePath = Path.Combine(_trainingDataDirectory, $"{baseName}_original.jpg");
            using (var stream = new FileStream(originalImagePath, FileMode.Create))
            {
                await originalImage.CopyToAsync(stream);
            }

            // 保存处理后的图片
            var processedImagePath = Path.Combine(_trainingDataDirectory, $"{baseName}_processed.jpg");
            using (var stream = new FileStream(processedImagePath, FileMode.Create))
            {
                await processedImage.CopyToAsync(stream);
            }

            // 解析角点坐标
            var cornerValues = corners.Split(',').Select(float.Parse).ToArray();
            if (cornerValues.Length != 8)
            {
                return BadRequest(new { success = false, message = "角点坐标格式错误，应为8个数值" });
            }

            // 保存元数据
            var metadata = new
            {
                timestamp = DateTimeOffset.UtcNow,
                originalImageFile = $"{baseName}_original.jpg",
                processedImageFile = $"{baseName}_processed.jpg",
                corners = new
                {
                    topLeft = new { x = cornerValues[0], y = cornerValues[1] },
                    topRight = new { x = cornerValues[2], y = cornerValues[3] },
                    bottomRight = new { x = cornerValues[4], y = cornerValues[5] },
                    bottomLeft = new { x = cornerValues[6], y = cornerValues[7] }
                },
                deviceInfo = new
                {
                    model = deviceModel,
                    platform = devicePlatform
                },
                originalImageSize = new FileInfo(originalImagePath).Length,
                processedImageSize = new FileInfo(processedImagePath).Length
            };

            var metadataPath = Path.Combine(_trainingDataDirectory, $"{baseName}.json");
            var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
            var jsonString = JsonSerializer.Serialize(metadata, jsonOptions);
            await System.IO.File.WriteAllTextAsync(metadataPath, jsonString);

            _logger.LogInformation($"训练数据已保存: {baseName}, 设备: {deviceModel} ({devicePlatform})");

            return Ok(new
            {
                success = true,
                message = "训练数据上传成功",
                trainingId = baseName
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "上传训练数据时发生错误");
            return StatusCode(500, new
            {
                success = false,
                message = $"上传失败: {ex.Message}"
            });
        }
    }

    /// <summary>
    /// 获取训练数据统计信息
    /// </summary>
    [HttpGet("stats")]
    public ActionResult<object> GetStats()
    {
        try
        {
            var directory = new DirectoryInfo(_trainingDataDirectory);

            if (!directory.Exists)
            {
                return Ok(new
                {
                    totalSamples = 0,
                    totalSize = 0L,
                    lastUpdate = (DateTime?)null
                });
            }

            var jsonFiles = directory.GetFiles("*.json");
            var totalSize = directory.GetFiles("*.jpg").Sum(f => f.Length);
            var lastUpdate = jsonFiles.Any() ? jsonFiles.Max(f => f.LastWriteTimeUtc) : (DateTime?)null;

            return Ok(new
            {
                totalSamples = jsonFiles.Length,
                totalSize = totalSize,
                lastUpdate = lastUpdate
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "获取训练数据统计时发生错误");
            return StatusCode(500, new { error = ex.Message });
        }
    }
}
