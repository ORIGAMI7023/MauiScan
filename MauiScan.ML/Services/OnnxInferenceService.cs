using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MauiScan.ML.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

#if WINDOWS
using OpenCvSharp;
#elif ANDROID || IOS || MACCATALYST
using MauiScan.ML.Interop;
#endif

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
    /// 检测图片中的四个角点（使用预处理的 RGB 数据）
    /// </summary>
    public async Task<MLDetectionResult> DetectCornersFromRgbAsync(float[] rgbData, int originalWidth, int originalHeight, byte[]? originalImageBytes = null)
    {
        System.Diagnostics.Debug.WriteLine($"[ML] DetectCornersFromRgbAsync started, original: {originalWidth}x{originalHeight}");

        await EnsureInitializedAsync();
        System.Diagnostics.Debug.WriteLine($"[ML] Model initialized");

        if (_session == null)
            throw new InvalidOperationException("模型未成功加载");

        return await Task.Run(() =>
        {
            System.Diagnostics.Debug.WriteLine($"[ML] Background task started");
            var sw = System.Diagnostics.Stopwatch.StartNew();

            // 直接从 float[] 构建 Tensor（CHW 格式）
            // ⚠️ 使用 Memory<float> 构造函数，直接使用现有数组避免复制
            System.Diagnostics.Debug.WriteLine($"[ML] Input rgbData length: {rgbData.Length}, expected: {3 * _config.InputHeight * _config.InputWidth}");
            System.Diagnostics.Debug.WriteLine($"[ML] rgbData stats - min: {rgbData.Min():F3}, max: {rgbData.Max():F3}, mean: {rgbData.Average():F3}");

            var inputTensor = new DenseTensor<float>(rgbData, new[] { 1, 3, _config.InputHeight, _config.InputWidth });
            System.Diagnostics.Debug.WriteLine($"[ML] Tensor created from RGB data, took {sw.ElapsedMilliseconds}ms");

            // 运行推理
            sw.Restart();
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            using var results = _session.Run(inputs);
            System.Diagnostics.Debug.WriteLine($"[ML] Inference completed, took {sw.ElapsedMilliseconds}ms");

            // 解析输出
            var coordinates = results.First(r => r.Name == "coordinates").AsEnumerable<float>().ToArray();
            System.Diagnostics.Debug.WriteLine($"[ML] Coordinates: [{string.Join(", ", coordinates.Select(c => c.ToString("F3")))}]");

            // 反归一化坐标到原始图片尺寸
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

            System.Diagnostics.Debug.WriteLine($"[ML] ML corners: TL({corners.TopLeftX:F1},{corners.TopLeftY:F1}) TR({corners.TopRightX:F1},{corners.TopRightY:F1}) BR({corners.BottomRightX:F1},{corners.BottomRightY:F1}) BL({corners.BottomLeftX:F1},{corners.BottomLeftY:F1})");

            // ⭐ Stage 2: CV 精修（如果提供了原图数据）
            QuadrilateralPoints finalCorners;
            if (originalImageBytes != null)
            {
                System.Diagnostics.Debug.WriteLine($"[ML] Starting Stage 2: Corner refinement");
                finalCorners = RefineCorners(corners, originalImageBytes, originalWidth, originalHeight);
                System.Diagnostics.Debug.WriteLine($"[ML] Refined corners: TL({finalCorners.TopLeftX:F1},{finalCorners.TopLeftY:F1}) TR({finalCorners.TopRightX:F1},{finalCorners.TopRightY:F1}) BR({finalCorners.BottomRightX:F1},{finalCorners.BottomRightY:F1}) BL({finalCorners.BottomLeftX:F1},{finalCorners.BottomLeftY:F1})");
            }
            else
            {
                System.Diagnostics.Debug.WriteLine($"[ML] No original image provided, skipping CV refinement");
                finalCorners = corners;
            }

            System.Diagnostics.Debug.WriteLine($"[ML] DetectCornersFromRgbAsync completed successfully");

            return new MLDetectionResult
            {
                Corners = finalCorners,
                MLRawCorners = corners,  // 保存 ML 原始输出（用于调试对比）
                NormalizedCoordinates = coordinates,  // 保存归一化坐标
                Confidence = 1.0f,
                CornerConfidences = new float[] { 1.0f, 1.0f, 1.0f, 1.0f }
            };
        });
    }

    /// <summary>
    /// 检测图片中的四个角点（带原始尺寸信息）
    /// </summary>
    public async Task<MLDetectionResult> DetectCornersAsync(byte[] imageBytes, int originalWidth, int originalHeight)
    {
        // 使用 ImageSharp 加载图片（兼容性方案）
        System.Diagnostics.Debug.WriteLine($"[ML] DetectCornersAsync started, image size: {imageBytes.Length} bytes, original: {originalWidth}x{originalHeight}");

        await EnsureInitializedAsync();

        if (_session == null)
            throw new InvalidOperationException("模型未成功加载");

        return await Task.Run(() =>
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            using var image = Image.Load<Rgb24>(imageBytes);
            System.Diagnostics.Debug.WriteLine($"[ML] Image loaded: {image.Width}x{image.Height}, took {sw.ElapsedMilliseconds}ms");

            if (image.Width != _config.InputWidth || image.Height != _config.InputHeight)
            {
                sw.Restart();
                image.Mutate(x => x.Resize(_config.InputWidth, _config.InputHeight));
                System.Diagnostics.Debug.WriteLine($"[ML] Image resized, took {sw.ElapsedMilliseconds}ms");
            }

            sw.Restart();
            var inputTensor = ImageToTensor(image);
            System.Diagnostics.Debug.WriteLine($"[ML] Tensor created, took {sw.ElapsedMilliseconds}ms");

            sw.Restart();
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            using var results = _session.Run(inputs);
            System.Diagnostics.Debug.WriteLine($"[ML] Inference completed, took {sw.ElapsedMilliseconds}ms");

            var coordinates = results.First(r => r.Name == "coordinates").AsEnumerable<float>().ToArray();
            System.Diagnostics.Debug.WriteLine($"[ML] Coordinates: [{string.Join(", ", coordinates.Select(c => c.ToString("F3")))}]");

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

            return new MLDetectionResult
            {
                Corners = corners,
                MLRawCorners = corners,  // 此方法无 CV 精修，两者相同
                NormalizedCoordinates = coordinates,
                Confidence = 1.0f,
                CornerConfidences = new float[] { 1.0f, 1.0f, 1.0f, 1.0f }
            };
        });
    }

    /// <summary>
    /// 检测图片中的四个角点（自动检测尺寸）
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

            // 1. 加载图片
            var sw = System.Diagnostics.Stopwatch.StartNew();
            System.Diagnostics.Debug.WriteLine($"[ML] Loading image with ImageSharp...");
            using var image = Image.Load<Rgb24>(imageBytes);
            var originalWidth = image.Width;
            var originalHeight = image.Height;
            System.Diagnostics.Debug.WriteLine($"[ML] Image loaded: {originalWidth}x{originalHeight}, took {sw.ElapsedMilliseconds}ms");

            // 2. 缩放到模型输入尺寸（使用 Bilinear 插值，与 Python PIL 一致）
            if (originalWidth != _config.InputWidth || originalHeight != _config.InputHeight)
            {
                sw.Restart();
                image.Mutate(x => x.Resize(new SixLabors.ImageSharp.Processing.ResizeOptions
                {
                    Size = new SixLabors.ImageSharp.Size(_config.InputWidth, _config.InputHeight),
                    Sampler = KnownResamplers.Triangle,  // Triangle = Bilinear
                    Mode = SixLabors.ImageSharp.Processing.ResizeMode.Stretch
                }));
                System.Diagnostics.Debug.WriteLine($"[ML] Image resized to {_config.InputWidth}x{_config.InputHeight} (Bilinear), took {sw.ElapsedMilliseconds}ms");
            }
            else
            {
                System.Diagnostics.Debug.WriteLine($"[ML] Image already at target size, skipping resize");
            }

            // 3. 转换为 ONNX 输入 Tensor (NCHW 格式: [1, 3, 512, 512])
            sw.Restart();
            var inputTensor = ImageToTensor(image);
            System.Diagnostics.Debug.WriteLine($"[ML] Tensor created, took {sw.ElapsedMilliseconds}ms");

            // 调试：打印像素统计信息
            var tensorData = inputTensor.ToArray();
            System.Diagnostics.Debug.WriteLine($"[ML] Tensor stats - min: {tensorData.Min():F3}, max: {tensorData.Max():F3}, mean: {tensorData.Average():F3}");
            // 打印前10个像素的RGB值（CHW格式，所以R在前512*512个）
            int pixelCount = _config.InputWidth * _config.InputHeight;
            var first10R = string.Join(", ", tensorData.Take(10).Select(v => v.ToString("F3")));
            var first10G = string.Join(", ", tensorData.Skip(pixelCount).Take(10).Select(v => v.ToString("F3")));
            var first10B = string.Join(", ", tensorData.Skip(2 * pixelCount).Take(10).Select(v => v.ToString("F3")));
            System.Diagnostics.Debug.WriteLine($"[ML] First 10 R: {first10R}");
            System.Diagnostics.Debug.WriteLine($"[ML] First 10 G: {first10G}");
            System.Diagnostics.Debug.WriteLine($"[ML] First 10 B: {first10B}");

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

            // 7. ⭐ Stage 2: 传统CV精修（在原图上精修到亚像素级）
            System.Diagnostics.Debug.WriteLine($"[ML] Starting Stage 2: Corner refinement");
            var refinedCorners = RefineCorners(corners, imageBytes, originalWidth, originalHeight);

            System.Diagnostics.Debug.WriteLine($"[ML] DetectCornersAsync completed successfully (with refinement)");

            return new MLDetectionResult
            {
                Corners = refinedCorners,
                MLRawCorners = corners,  // 保存 ML 原始输出（用于调试对比）
                NormalizedCoordinates = coordinates,  // 保存归一化坐标
                Confidence = 1.0f,
                CornerConfidences = new float[] { 1.0f, 1.0f, 1.0f, 1.0f }
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

    #region Corner Refinement (Stage 2: Traditional CV)

#if WINDOWS
    /// <summary>
    /// 精修ML预测的角点（使用传统CV方法达到亚像素精度）
    /// </summary>
    /// <param name="mlCorners">ML预测的粗略角点（归一化坐标）</param>
    /// <param name="imageBytes">原始图片数据</param>
    /// <param name="originalWidth">原图宽度</param>
    /// <param name="originalHeight">原图高度</param>
    /// <returns>精修后的角点</returns>
    private QuadrilateralPoints RefineCorners(
        QuadrilateralPoints mlCorners,
        byte[] imageBytes,
        int originalWidth,
        int originalHeight)
    {
        System.Diagnostics.Debug.WriteLine($"[Refinement] Starting corner refinement on {originalWidth}x{originalHeight} image");

        // 加载原图为灰度图
        using var mat = Mat.FromImageData(imageBytes, ImreadModes.Grayscale);

        if (mat.Empty())
        {
            System.Diagnostics.Debug.WriteLine($"[Refinement] Failed to load image, returning ML corners");
            return mlCorners;
        }

        try
        {
            // 对4个角点分别精修
            var refined = new QuadrilateralPoints
            {
                TopLeftX = mlCorners.TopLeftX,
                TopLeftY = mlCorners.TopLeftY,
                TopRightX = mlCorners.TopRightX,
                TopRightY = mlCorners.TopRightY,
                BottomRightX = mlCorners.BottomRightX,
                BottomRightY = mlCorners.BottomRightY,
                BottomLeftX = mlCorners.BottomLeftX,
                BottomLeftY = mlCorners.BottomLeftY
            };

            // 精修左上角
            var (tlX, tlY) = RefineSingleCorner(mat, mlCorners.TopLeftX, mlCorners.TopLeftY);
            if (tlX >= 0)
            {
                refined.TopLeftX = tlX;
                refined.TopLeftY = tlY;
            }

            // 精修右上角
            var (trX, trY) = RefineSingleCorner(mat, mlCorners.TopRightX, mlCorners.TopRightY);
            if (trX >= 0)
            {
                refined.TopRightX = trX;
                refined.TopRightY = trY;
            }

            // 精修右下角
            var (brX, brY) = RefineSingleCorner(mat, mlCorners.BottomRightX, mlCorners.BottomRightY);
            if (brX >= 0)
            {
                refined.BottomRightX = brX;
                refined.BottomRightY = brY;
            }

            // 精修左下角
            var (blX, blY) = RefineSingleCorner(mat, mlCorners.BottomLeftX, mlCorners.BottomLeftY);
            if (blX >= 0)
            {
                refined.BottomLeftX = blX;
                refined.BottomLeftY = blY;
            }

            System.Diagnostics.Debug.WriteLine($"[Refinement] Refinement completed successfully");
            return refined;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[Refinement] Error during refinement: {ex.Message}");
            // 精修失败，返回ML原始结果
            return mlCorners;
        }
    }

    /// <summary>
    /// 精修单个角点
    /// </summary>
    /// <param name="image">灰度图</param>
    /// <param name="mlX">ML预测的X坐标</param>
    /// <param name="mlY">ML预测的Y坐标</param>
    /// <returns>精修后的坐标，失败返回(-1, -1)</returns>
    private (float x, float y) RefineSingleCorner(Mat image, float mlX, float mlY)
    {
        const int patchSize = 64; // 搜索窗口大小

        try
        {
            // 1. 裁剪patch（确保不越界）
            int centerX = (int)mlX;
            int centerY = (int)mlY;
            int halfPatch = patchSize / 2;

            int x1 = Math.Max(0, centerX - halfPatch);
            int y1 = Math.Max(0, centerY - halfPatch);
            int x2 = Math.Min(image.Width, centerX + halfPatch);
            int y2 = Math.Min(image.Height, centerY + halfPatch);

            if (x2 - x1 < 20 || y2 - y1 < 20)
            {
                // Patch太小，返回失败
                return (-1, -1);
            }

            var roi = new Rect(x1, y1, x2 - x1, y2 - y1);
            using var patch = new Mat(image, roi);

            // 2. Canny边缘检测
            using var edges = new Mat();
            Cv2.Canny(patch, edges, 50, 150, 3);

            // 3. 霍夫直线检测
            var lines = Cv2.HoughLinesP(
                edges,
                rho: 1,
                theta: Math.PI / 180,
                threshold: 30,
                minLineLength: 20,
                maxLineGap: 5
            );

            if (lines == null || lines.Length < 2)
            {
                // 检测到的直线太少
                return (-1, -1);
            }

            // 4. 直线聚类（水平 vs 垂直）
            var horizontalLines = new List<LineSegmentPoint>();
            var verticalLines = new List<LineSegmentPoint>();

            foreach (var line in lines)
            {
                float dx = Math.Abs(line.P2.X - line.P1.X);
                float dy = Math.Abs(line.P2.Y - line.P1.Y);

                if (dx > dy) // 更水平
                {
                    horizontalLines.Add(line);
                }
                else // 更垂直
                {
                    verticalLines.Add(line);
                }
            }

            if (horizontalLines.Count == 0 || verticalLines.Count == 0)
            {
                // 没有找到两组直线
                return (-1, -1);
            }

            // 5. 拟合直线（简化版：使用中位数斜率和截距）
            var hLine = FitLine(horizontalLines);
            var vLine = FitLine(verticalLines);

            if (hLine == null || vLine == null)
            {
                return (-1, -1);
            }

            // 6. 计算交点
            var intersection = ComputeIntersection(hLine.Value, vLine.Value);
            if (intersection == null)
            {
                return (-1, -1);
            }

            // 7. 转换回原图坐标
            float refinedX = x1 + intersection.Value.X;
            float refinedY = y1 + intersection.Value.Y;

            // 8. 验证精修结果是否合理（距离ML预测不能太远）
            float distance = (float)Math.Sqrt(
                Math.Pow(refinedX - mlX, 2) + Math.Pow(refinedY - mlY, 2)
            );

            if (distance > patchSize) // 超出搜索范围
            {
                return (-1, -1);
            }

            return (refinedX, refinedY);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[Refinement] Error in RefineSingleCorner: {ex.Message}");
            return (-1, -1);
        }
    }

    /// <summary>
    /// 拟合直线（返回直线方程参数）
    /// </summary>
    private (float k, float b)? FitLine(List<LineSegmentPoint> lines)
    {
        if (lines.Count == 0) return null;

        // 简化版：使用所有线段端点进行最小二乘拟合
        var points = new List<Point2f>();
        foreach (var line in lines)
        {
            points.Add(new Point2f(line.P1.X, line.P1.Y));
            points.Add(new Point2f(line.P2.X, line.P2.Y));
        }

        // 计算平均值
        float avgX = points.Average(p => p.X);
        float avgY = points.Average(p => p.Y);

        // 最小二乘法
        float numerator = 0;
        float denominator = 0;

        foreach (var p in points)
        {
            numerator += (p.X - avgX) * (p.Y - avgY);
            denominator += (p.X - avgX) * (p.X - avgX);
        }

        if (Math.Abs(denominator) < 1e-6)
        {
            // 垂直线，无法用 y=kx+b 表示
            return null;
        }

        float k = numerator / denominator;
        float b = avgY - k * avgX;

        return (k, b);
    }

    /// <summary>
    /// 计算两条直线的交点
    /// </summary>
    private Point2f? ComputeIntersection((float k, float b) line1, (float k, float b) line2)
    {
        // line1: y = k1*x + b1
        // line2: y = k2*x + b2
        // 交点: k1*x + b1 = k2*x + b2

        float k1 = line1.k, b1 = line1.b;
        float k2 = line2.k, b2 = line2.b;

        if (Math.Abs(k1 - k2) < 1e-6)
        {
            // 平行线
            return null;
        }

        float x = (b2 - b1) / (k1 - k2);
        float y = k1 * x + b1;

        return new Point2f(x, y);
    }

#elif ANDROID || IOS || MACCATALYST
    /// <summary>
    /// 精修ML预测的角点（使用 Native OpenCV）
    /// </summary>
    private QuadrilateralPoints RefineCorners(
        QuadrilateralPoints mlCorners,
        byte[] imageBytes,
        int originalWidth,
        int originalHeight)
    {
        System.Diagnostics.Debug.WriteLine($"[Refinement] Starting Native OpenCV corner refinement on {originalWidth}x{originalHeight} image");

        try
        {
            var refined = new QuadrilateralPoints
            {
                TopLeftX = mlCorners.TopLeftX,
                TopLeftY = mlCorners.TopLeftY,
                TopRightX = mlCorners.TopRightX,
                TopRightY = mlCorners.TopRightY,
                BottomRightX = mlCorners.BottomRightX,
                BottomRightY = mlCorners.BottomRightY,
                BottomLeftX = mlCorners.BottomLeftX,
                BottomLeftY = mlCorners.BottomLeftY
            };

            // 精修4个角点
            int success = NativeOpenCv.RefineCorner(imageBytes, imageBytes.Length,
                mlCorners.TopLeftX, mlCorners.TopLeftY, out float tlX, out float tlY);
            if (success == 1) { refined.TopLeftX = tlX; refined.TopLeftY = tlY; }

            success = NativeOpenCv.RefineCorner(imageBytes, imageBytes.Length,
                mlCorners.TopRightX, mlCorners.TopRightY, out float trX, out float trY);
            if (success == 1) { refined.TopRightX = trX; refined.TopRightY = trY; }

            success = NativeOpenCv.RefineCorner(imageBytes, imageBytes.Length,
                mlCorners.BottomRightX, mlCorners.BottomRightY, out float brX, out float brY);
            if (success == 1) { refined.BottomRightX = brX; refined.BottomRightY = brY; }

            success = NativeOpenCv.RefineCorner(imageBytes, imageBytes.Length,
                mlCorners.BottomLeftX, mlCorners.BottomLeftY, out float blX, out float blY);
            if (success == 1) { refined.BottomLeftX = blX; refined.BottomLeftY = blY; }

            System.Diagnostics.Debug.WriteLine($"[Refinement] Native OpenCV refinement completed successfully");
            return refined;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[Refinement] Native OpenCV refinement failed: {ex.Message}");
            return mlCorners;
        }
    }
#else
    /// <summary>
    /// 精修ML预测的角点（其他平台降级实现）
    /// </summary>
    private QuadrilateralPoints RefineCorners(
        QuadrilateralPoints mlCorners,
        byte[] imageBytes,
        int originalWidth,
        int originalHeight)
    {
        System.Diagnostics.Debug.WriteLine($"[Refinement] Corner refinement not available on this platform, returning ML corners");
        return mlCorners;
    }
#endif

    #endregion

    public void Dispose()
    {
        _session?.Dispose();
        _initLock?.Dispose();
    }
}
