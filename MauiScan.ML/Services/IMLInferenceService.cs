using MauiScan.ML.Models;

namespace MauiScan.ML.Services;

/// <summary>
/// ML 推理服务接口
/// </summary>
public interface IMLInferenceService
{
    /// <summary>
    /// 检测图片中的 PPT 四个角点
    /// </summary>
    /// <param name="imageBytes">原始图片字节数据（JPG/PNG）</param>
    /// <returns>检测结果，包含角点坐标和置信度</returns>
    Task<MLDetectionResult> DetectCornersAsync(byte[] imageBytes);

    /// <summary>
    /// 检查 ML 模型是否可用
    /// </summary>
    /// <returns>true 如果模型文件存在且可加载</returns>
    Task<bool> IsModelAvailableAsync();

    /// <summary>
    /// 获取模型信息
    /// </summary>
    /// <returns>模型元数据（版本、训练信息等）</returns>
    Task<ModelInfo> GetModelInfoAsync();

    /// <summary>
    /// 预热模型（首次加载时可能较慢，预热后加快后续推理）
    /// </summary>
    Task WarmupAsync();
}
