namespace MauiScan.ML.Models;

/// <summary>
/// ML 模型配置
/// </summary>
public class ModelConfig
{
    /// <summary>
    /// 模型输入图片的宽度（像素）
    /// </summary>
    public int InputWidth { get; set; } = 512;

    /// <summary>
    /// 模型输入图片的高度（像素）
    /// </summary>
    public int InputHeight { get; set; } = 512;

    /// <summary>
    /// 模型输入的通道数（RGB = 3）
    /// </summary>
    public int InputChannels { get; set; } = 3;

    /// <summary>
    /// 高质量置信度阈值（>= 此值时直接使用 ML 结果）
    /// </summary>
    public float HighQualityThreshold { get; set; } = 0.85f;

    /// <summary>
    /// 中等质量置信度阈值（>= 此值时使用 ML + 传统算法融合）
    /// </summary>
    public float MediumQualityThreshold { get; set; } = 0.60f;

    /// <summary>
    /// 是否启用 GPU 加速（Android NNAPI / iOS CoreML）
    /// </summary>
    public bool EnableGpuAcceleration { get; set; } = true;

    /// <summary>
    /// 推理超时时间（毫秒）
    /// </summary>
    public int InferenceTimeoutMs { get; set; } = 5000;
}

/// <summary>
/// 模型信息
/// </summary>
public class ModelInfo
{
    /// <summary>
    /// 模型版本号
    /// </summary>
    public string Version { get; set; } = "1.0.0";

    /// <summary>
    /// 模型训练日期
    /// </summary>
    public DateTime TrainedDate { get; set; }

    /// <summary>
    /// 训练样本数量
    /// </summary>
    public int TrainingSamples { get; set; }

    /// <summary>
    /// 验证集准确率（0-1）
    /// </summary>
    public float ValidationAccuracy { get; set; }

    /// <summary>
    /// 模型文件大小（字节）
    /// </summary>
    public long FileSizeBytes { get; set; }

    /// <summary>
    /// 模型描述
    /// </summary>
    public string Description { get; set; } = "PPT 四角点检测模型";
}
