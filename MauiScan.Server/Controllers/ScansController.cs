using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.SignalR;
using MauiScan.Server.Hubs;
using MauiScan.Server.Models;
using MauiScan.Server.Services;

namespace MauiScan.Server.Controllers;

[ApiController]
[Route("api/[controller]")]
public class ScansController : ControllerBase
{
    private readonly IFileStorageService _fileStorage;
    private readonly IHubContext<ScanHub> _hubContext;
    private readonly ILogger<ScansController> _logger;

    public ScansController(
        IFileStorageService fileStorage,
        IHubContext<ScanHub> hubContext,
        ILogger<ScansController> logger)
    {
        _fileStorage = fileStorage;
        _hubContext = hubContext;
        _logger = logger;
    }

    /// <summary>
    /// 上传扫描图片
    /// </summary>
    [HttpPost("upload")]
    [RequestSizeLimit(20 * 1024 * 1024)] // 20MB
    public async Task<ActionResult<UploadResponse>> Upload(
        [FromForm] IFormFile file,
        [FromForm] int width = 0,
        [FromForm] int height = 0)
    {
        try
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest(new UploadResponse
                {
                    Success = false,
                    Message = "文件为空"
                });
            }

            // 验证文件类型
            if (!file.ContentType.StartsWith("image/"))
            {
                return BadRequest(new UploadResponse
                {
                    Success = false,
                    Message = "文件类型必须是图片"
                });
            }

            // 保存文件
            var scanImage = await _fileStorage.SaveFileAsync(file, width, height);

            _logger.LogInformation($"文件已上传: {scanImage.FileName}, 大小: {scanImage.FileSize} bytes");

            // 通过 SignalR 广播给所有连接的设备
            await _hubContext.Clients.Group("AllDevices")
                .SendAsync("ReceiveNewScan", scanImage);

            _logger.LogInformation($"已广播新扫描到所有设备: {scanImage.FileName}");

            return Ok(new UploadResponse
            {
                Success = true,
                Message = "上传成功",
                ScanImage = scanImage
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "上传文件时发生错误");
            return StatusCode(500, new UploadResponse
            {
                Success = false,
                Message = $"上传失败: {ex.Message}"
            });
        }
    }

    /// <summary>
    /// 获取最近的扫描列表
    /// </summary>
    [HttpGet("recent")]
    public async Task<ActionResult<List<ScanImageDto>>> GetRecent([FromQuery] int limit = 10)
    {
        try
        {
            var scans = await _fileStorage.GetRecentScansAsync(limit);
            return Ok(scans);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "获取最近扫描列表时发生错误");
            return StatusCode(500, new List<ScanImageDto>());
        }
    }

    /// <summary>
    /// 下载扫描图片
    /// </summary>
    [HttpGet("{fileName}")]
    public async Task<IActionResult> Download(string fileName)
    {
        try
        {
            var result = await _fileStorage.GetFileAsync(fileName);

            if (result == null)
            {
                return NotFound();
            }

            return File(result.Value.fileData, result.Value.contentType, fileName);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"下载文件时发生错误: {fileName}");
            return StatusCode(500);
        }
    }
}
