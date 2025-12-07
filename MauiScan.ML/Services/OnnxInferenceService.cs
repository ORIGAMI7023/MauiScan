using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MauiScan.ML.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace MauiScan.ML.Services;

/// <summary>
/// ONNX Runtime 推理服务实现
/// </summary>
public class OnnxInferenceService : IMLInferenceService, IDisposable
{
    private readonly ModelConfig _config;
    private readonly string _modelPath;
    private InferenceSession? _session;
    private bool _isInitialized;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    public OnnxInferenceService(string modelPath, ModelConfig? config = null)
    {
        _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
        _config = config ?? new ModelConfig();
    }

    /// <summary>
    /// 检测图片中的四个角点
    /// </summary>
    public async Task<MLDetectionResult> DetectCornersAsync(byte[] imageBytes)
    {
        System.Diagnostics.Debug.WriteLine($"[ML] DetectCornersAsync started, image size: {imageBytes.Length} bytes");

        await EnsureInitializedAsync();
        System.Diagnostics.Debug.WriteLine($"[ML] Model initialized");

        if (_session == null)
            throw new InvalidOperationException("模型未成功加载");

        // 将整个推理过程放到后台线程
        return await Task.Run(() =>
        {
            System.Diagnostics.Debug.WriteLine($"[ML] Background task started");

            // 1. 加载和预处理图片
            var sw = System.Diagnostics.Stopwatch.StartNew();
            using var image = Image.Load<Rgb24>(imageBytes);
            var originalWidth = image.Width;
            var originalHeight = image.Height;
            System.Diagnostics.Debug.WriteLine($"[ML] Image loaded: {originalWidth}x{originalHeight}, took {sw.ElapsedMilliseconds}ms");

            // 2. 缩放到模型输入尺寸（如果需要）
            if (originalWidth != _config.InputWidth || originalHeight != _config.InputHeight)
            {
                sw.Restart();
                image.Mutate(x => x.Resize(_config.InputWidth, _config.InputHeight));
                System.Diagnostics.Debug.WriteLine($"[ML] Image resized to {_config.InputWidth}x{_config.InputHeight}, took {sw.ElapsedMilliseconds}ms");
            }
            else
            {
                System.Diagnostics.Debug.WriteLine($"[ML] Image already at target size, skipping resize");
            }

            // 3. 转换为 ONNX 输入 Tensor (NCHW 格式: [1, 3, 512, 512])
            sw.Restart();
            var inputTensor = ImageToTensor(image);
            System.Diagnostics.Debug.WriteLine($"[ML] Tensor created, took {sw.ElapsedMilliseconds}ms");

            // 4. 运行推理
            sw.Restart();
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            using var results = _session.Run(inputs);
            System.Diagnostics.Debug.WriteLine($"[ML] Inference completed, took {sw.ElapsedMilliseconds}ms");

            // 5. 解析输出
            sw.Restart();
            var coordinates = results.First(r => r.Name == "coordinates").AsEnumerable<float>().ToArray();
            // 模型输出的置信度未经训练，暂时忽略
            // var modelConfidence = results.First(r => r.Name == "confidence").AsEnumerable<float>().First();
            System.Diagnostics.Debug.WriteLine($"[ML] Output parsed, took {sw.ElapsedMilliseconds}ms");
            System.Diagnostics.Debug.WriteLine($"[ML] Coordinates: [{string.Join(", ", coordinates.Select(c => c.ToString("F3")))}]");

            // 6. 反归一化坐标到原始图片尺寸
            var corners = new QuadrilateralPoints
            {
                TopLeftX = coordinates[0] * originalWidth,
                TopLeftY = coordinates[1] * originalHeight,
                TopRightX = coordinates[2] * originalWidth,
                TopRightY = coordinates[3] * originalHeight,
                BottomRightX = coordinates[4] * originalWidth,
                BottomRightY = coordinates[5] * originalHeight,
                BottomLeftX = coordinates[6] * originalWidth,
                BottomLeftY = coordinates[7] * originalHeight
            };

            // 7. 基于坐标合理性计算置信度（因为模型置信度头未训练）
            // 检查：四边形是否合理（顺时针顺序、面积合理、角度合理）
            float confidence = CalculateConfidenceFromCoordinates(coordinates);
            System.Diagnostics.Debug.WriteLine($"[ML] Calculated confidence: {confidence:F3}");

            var cornerConfidences = new float[4];
            for (int i = 0; i < 4; i++)
            {
                cornerConfidences[i] = confidence;
            }

            System.Diagnostics.Debug.WriteLine($"[ML] DetectCornersAsync completed successfully");

            return new MLDetectionResult
            {
                Corners = corners,
                Confidence = confidence,
                CornerConfidences = cornerConfidences
            };
        });
    }

    /// <summary>
    /// 检查模型是否可用
    /// </summary>
    public Task<bool> IsModelAvailableAsync()
    {
        return Task.FromResult(File.Exists(_modelPath));
    }

    /// <summary>
    /// 获取模型信息
    /// </summary>
    public async Task<ModelInfo> GetModelInfoAsync()
    {
        if (!await IsModelAvailableAsync())
        {
            throw new FileNotFoundException("模型文件不存在", _modelPath);
        }

        var fileInfo = new FileInfo(_modelPath);

        // 从模型文件或配置文件读取元数据
        // 这里使用默认值，实际应从模型元数据或外部配置读取
        return new ModelInfo
        {
            Version = "1.0.0",
            TrainedDate = fileInfo.LastWriteTime,
            TrainingSamples = 0, // TODO: 从元数据读取
            ValidationAccuracy = 0.0f, // TODO: 从元数据读取
            FileSizeBytes = fileInfo.Length,
            Description = "PPT 四角点检测模型 (ONNX)"
        };
    }

    /// <summary>
    /// 预热模型（使用虚拟输入运行一次推理）
    /// </summary>
    public async Task WarmupAsync()
    {
        await EnsureInitializedAsync();

        // 创建一个虚拟的黑色图片进行预热
        var dummyImage = new byte[_config.InputWidth * _config.InputHeight * 3];

        using var image = Image.LoadPixelData<Rgb24>(dummyImage, _config.InputWidth, _config.InputHeight);

        // 转换为 JPEG 字节流
        using var ms = new MemoryStream();
        await image.SaveAsJpegAsync(ms);

        // 运行一次推理（忽略结果）
        _ = await DetectCornersAsync(ms.ToArray());
    }

    /// <summary>
    /// 确保模型已初始化
    /// </summary>
    private async Task EnsureInitializedAsync()
    {
        if (_isInitialized)
            return;

        await _initLock.WaitAsync();
        try
        {
            if (_isInitialized)
                return;

            System.Diagnostics.Debug.WriteLine($"[ML] Starting model initialization...");
            System.Diagnostics.Debug.WriteLine($"[ML] Model path: {_modelPath}");

            if (!await IsModelAvailableAsync())
            {
                System.Diagnostics.Debug.WriteLine($"[ML] ERROR: Model file not found!");
                throw new FileNotFoundException("模型文件不存在", _modelPath);
            }

            var fileInfo = new FileInfo(_modelPath);
            System.Diagnostics.Debug.WriteLine($"[ML] Model file found, size: {fileInfo.Length / (1024.0 * 1024.0):F2} MB");

            // 创建 SessionOptions
            var sessionOptions = new SessionOptions();
            System.Diagnostics.Debug.WriteLine($"[ML] SessionOptions created");

            // 启用 GPU 加速（如果配置启用）
            if (_config.EnableGpuAcceleration)
            {
                System.Diagnostics.Debug.WriteLine($"[ML] GPU acceleration requested");
                // Android: 尝试使用 NNAPI
                // iOS: 尝试使用 CoreML
                // Windows: 尝试使用 DirectML
                try
                {
                    // sessionOptions.AppendExecutionProvider_Nnapi(); // Android
                    // sessionOptions.AppendExecutionProvider_CoreML(); // iOS
                    // 默认使用 CPU，GPU 加速需要额外配置
                }
                catch
                {
                    // 如果 GPU 加速不可用，降级到 CPU
                }
            }

            // 加载模型
            var sw = System.Diagnostics.Stopwatch.StartNew();
            System.Diagnostics.Debug.WriteLine($"[ML] Loading ONNX model...");
            _session = new InferenceSession(_modelPath, sessionOptions);
            sw.Stop();
            System.Diagnostics.Debug.WriteLine($"[ML] Model loaded successfully in {sw.ElapsedMilliseconds}ms");

            _isInitialized = true;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[ML] ERROR during initialization: {ex.GetType().Name}");
            System.Diagnostics.Debug.WriteLine($"[ML] ERROR message: {ex.Message}");
            System.Diagnostics.Debug.WriteLine($"[ML] ERROR stack: {ex.StackTrace}");
            throw;
        }
        finally
        {
            _initLock.Release();
        }
    }

    /// <summary>
    /// 基于坐标合理性计算置信度
    /// 检查四边形是否符合 PPT 投影的几何特征
    /// </summary>
    private float CalculateConfidenceFromCoordinates(float[] coords)
    {
        // coords: [x1, y1, x2, y2, x3, y3, x4, y4]
        // 顺序: 左上(0), 右上(1), 右下(2), 左下(3)

        float score = 1.0f;

        // 1. 检查坐标是否在有效范围 [0, 1] 内
        for (int i = 0; i < 8; i++)
        {
            if (coords[i] < 0 || coords[i] > 1)
            {
                score *= 0.5f; // 超出范围，降低置信度
            }
        }

        // 2. 检查 X 坐标顺序：左 < 右
        float leftX = (coords[0] + coords[6]) / 2;   // 左上和左下的平均 X
        float rightX = (coords[2] + coords[4]) / 2;  // 右上和右下的平均 X
        if (leftX >= rightX)
        {
            score *= 0.3f; // X 顺序错误
        }

        // 3. 检查 Y 坐标顺序：上 < 下
        float topY = (coords[1] + coords[3]) / 2;    // 左上和右上的平均 Y
        float bottomY = (coords[5] + coords[7]) / 2; // 右下和左下的平均 Y
        if (topY >= bottomY)
        {
            score *= 0.3f; // Y 顺序错误
        }

        // 4. 检查四边形面积是否合理（至少占图片 5%，不超过 95%）
        float width = rightX - leftX;
        float height = bottomY - topY;
        float area = width * height;

        if (area < 0.05f)
        {
            score *= 0.5f; // 太小
        }
        else if (area > 0.95f)
        {
            score *= 0.8f; // 太大（可能是整张图）
        }

        // 5. 检查宽高比是否合理（PPT 通常是 4:3 或 16:9）
        if (width > 0 && height > 0)
        {
            float aspectRatio = width / height;
            // 合理范围：0.5 到 3.0
            if (aspectRatio < 0.5f || aspectRatio > 3.0f)
            {
                score *= 0.7f;
            }
        }

        return Math.Clamp(score, 0.0f, 1.0f);
    }

    /// <summary>
    /// 将图片转换为 ONNX Tensor (NCHW 格式)
    /// </summary>
    private DenseTensor<float> ImageToTensor(Image<Rgb24> image)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, _config.InputHeight, _config.InputWidth });

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < _config.InputHeight; y++)
            {
                var pixelRow = accessor.GetRowSpan(y);
                for (int x = 0; x < _config.InputWidth; x++)
                {
                    var pixel = pixelRow[x];

                    // 归一化到 [0, 1] 并按 CHW 格式填充
                    tensor[0, 0, y, x] = pixel.R / 255f; // R 通道
                    tensor[0, 1, y, x] = pixel.G / 255f; // G 通道
                    tensor[0, 2, y, x] = pixel.B / 255f; // B 通道
                }
            }
        });

        return tensor;
    }

    public void Dispose()
    {
        _session?.Dispose();
        _initLock?.Dispose();
    }
}
