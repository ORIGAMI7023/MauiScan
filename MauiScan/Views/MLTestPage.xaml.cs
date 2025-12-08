using MauiScan.ML.Services;
using MauiScan.ML.Models;
using System.Diagnostics;

namespace MauiScan.Views;

public partial class MLTestPage : ContentPage
{
    private readonly IMLInferenceService _mlService;
    private byte[]? _currentImageBytes;      // 512x512 缩放后的图片（用于显示）
    private byte[]? _originalImageBytes;     // 原始图片（用于透视变换和CV精修）
    private float[]? _cachedRgbData;         // 缓存的 RGB 数据（避免 JPEG 压缩损失）
    private int _originalWidth;              // 原始图片宽度
    private int _originalHeight;             // 原始图片高度
    private string? _lastErrorMessage;

    // test.jpg 的真实角点坐标（用于计算误差）
    private static readonly QuadrilateralPoints? TestGroundTruth = new QuadrilateralPoints
    {
        TopLeftX = 1056,
        TopLeftY = 424,
        TopRightX = 3261,
        TopRightY = 943,
        BottomRightX = 3461,
        BottomRightY = 2677,
        BottomLeftX = 979,
        BottomLeftY = 2656
    };
    private const int TestImageWidth = 4080;  // test.jpg 的原始宽度
    private const int TestImageHeight = 3060; // test.jpg 的原始高度

    public bool HasImage => _currentImageBytes != null;
    public bool HasResult { get; private set; }
    public bool HasError { get; private set; }

    public MLTestPage(IMLInferenceService mlService)
    {
        InitializeComponent();
        _mlService = mlService;
        BindingContext = this;
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();

        // 确保模型文件已复制
        ModelStatusLabel.Text = "正在检查模型文件...";
        try
        {
            await App.EnsureModelFileCopiedAsync();
            await CheckModelStatusAsync();

            HasError = false;
            OnPropertyChanged(nameof(HasError));
        }
        catch (Exception ex)
        {
            _lastErrorMessage = $"模型文件复制失败\n\n错误类型: {ex.GetType().Name}\n错误消息: {ex.Message}\n\n完整堆栈:\n{ex.StackTrace}";
            ModelStatusLabel.Text = $"❌ 模型文件复制失败: {ex.Message}";
            ModelStatusLabel.TextColor = Colors.Red;
            Debug.WriteLine($"[ML Test] Error ensuring model file: {ex}");

            HasError = true;
            OnPropertyChanged(nameof(HasError));
        }
    }

    private async Task CheckModelStatusAsync()
    {
        try
        {
            var isAvailable = await _mlService.IsModelAvailableAsync();

            if (isAvailable)
            {
                ModelStatusLabel.Text = "✅ 模型已加载";
                ModelStatusLabel.TextColor = Colors.Green;

                // 尝试获取模型信息
                try
                {
                    var modelInfo = await _mlService.GetModelInfoAsync();
                    ModelStatusLabel.Text += $" ({modelInfo.FileSizeBytes / (1024.0 * 1024.0):F2} MB)";
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[ML Test] Failed to get model info: {ex.Message}");
                }
            }
            else
            {
                var modelPath = Path.Combine(FileSystem.AppDataDirectory, "ppt_corner_detector.onnx");
                ModelStatusLabel.Text = $"❌ 模型文件不存在\n路径: {modelPath}";
                ModelStatusLabel.TextColor = Colors.Red;
                Debug.WriteLine($"[ML Test] Model file not found at: {modelPath}");
            }
        }
        catch (Exception ex)
        {
            ModelStatusLabel.Text = $"❌ 检查模型失败: {ex.Message}";
            ModelStatusLabel.TextColor = Colors.Red;
            Debug.WriteLine($"[ML Test] Error checking model: {ex}");
        }
    }

    private async void OnTakePhotoClicked(object sender, EventArgs e)
    {
        try
        {
            if (MediaPicker.Default.IsCaptureSupported)
            {
                var photo = await MediaPicker.Default.CapturePhotoAsync();

                if (photo != null)
                {
                    await LoadImageAsync(photo);
                    Debug.WriteLine($"[ML Test] Photo captured: {photo.FileName}");
                }
            }
            else
            {
                await DisplayAlert("不支持", "当前设备不支持拍照功能", "确定");
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"拍照失败: {ex.Message}", "确定");
        }
    }

    private async void OnSelectImageClicked(object sender, EventArgs e)
    {
        try
        {
            var result = await FilePicker.PickAsync(new PickOptions
            {
                PickerTitle = "选择一张 PPT 图片",
                FileTypes = FilePickerFileType.Images
            });

            if (result != null)
            {
                await LoadImageAsync(result);
                Debug.WriteLine($"[ML Test] Image loaded: {result.FileName}");
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"无法加载图片: {ex.Message}", "确定");
        }
    }

    private async Task LoadImageAsync(FileResult fileResult)
    {
        Debug.WriteLine($"[ML Test] Loading image: {fileResult.FileName}");

        // 读取原始图片
        using var stream = await fileResult.OpenReadAsync();
        using var originalStream = new MemoryStream();
        await stream.CopyToAsync(originalStream);

        // 保存原始图片字节数据
        _originalImageBytes = originalStream.ToArray();
        originalStream.Position = 0;

        Debug.WriteLine($"[ML Test] Original image size: {originalStream.Length / 1024.0:F1} KB");

        // 直接缩小到 ML 模型的输入尺寸 512x512
        const int targetSize = 512;

        try
        {
#if ANDROID
            // Android: 直接从 Bitmap 提取 RGB 数据，避免 JPEG 压缩损失
            _currentImageBytes = await Task.Run(() =>
            {
                using var bitmap = Android.Graphics.BitmapFactory.DecodeStream(originalStream);
                if (bitmap == null)
                    throw new Exception("无法解码图片");

                _originalWidth = bitmap.Width;
                _originalHeight = bitmap.Height;
                Debug.WriteLine($"[ML Test] Original dimensions: {bitmap.Width}x{bitmap.Height}");

                // 缩放到 512x512（使用 Canvas + Matrix 实现精确的双线性插值）
                Debug.WriteLine($"[ML Test] Resizing to: {targetSize}x{targetSize}");
                var sw = System.Diagnostics.Stopwatch.StartNew();

                // 创建目标 Bitmap
                using var scaledBitmap = Android.Graphics.Bitmap.CreateBitmap(targetSize, targetSize, Android.Graphics.Bitmap.Config.Argb8888!);
                using var canvas = new Android.Graphics.Canvas(scaledBitmap);

                // 使用 Matrix 缩放
                var matrix = new Android.Graphics.Matrix();
                float scaleX = (float)targetSize / bitmap.Width;
                float scaleY = (float)targetSize / bitmap.Height;
                matrix.SetScale(scaleX, scaleY);

                // 使用双线性插值绘制
                using var paint = new Android.Graphics.Paint();
                paint.FilterBitmap = true;  // 启用双线性过滤
                paint.AntiAlias = true;
                canvas.DrawBitmap(bitmap, matrix, paint);

                Debug.WriteLine($"[ML Test] Android resize took: {sw.ElapsedMilliseconds}ms");

                // ⚠️ 直接从 Bitmap 提取 RGB 数据，不经过 JPEG 压缩
                sw.Restart();
                int[] pixels = new int[targetSize * targetSize];
                scaledBitmap.GetPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize);

                _cachedRgbData = new float[3 * targetSize * targetSize];
                for (int y = 0; y < targetSize; y++)
                {
                    for (int x = 0; x < targetSize; x++)
                    {
                        int pixel = pixels[y * targetSize + x];
                        int r = (pixel >> 16) & 0xFF;
                        int g = (pixel >> 8) & 0xFF;
                        int b = pixel & 0xFF;

                        int idx = y * targetSize + x;
                        _cachedRgbData[idx] = r / 255f;
                        _cachedRgbData[targetSize * targetSize + idx] = g / 255f;
                        _cachedRgbData[2 * targetSize * targetSize + idx] = b / 255f;
                    }
                }
                Debug.WriteLine($"[ML Test] RGB data extracted: {_cachedRgbData.Length} floats, took {sw.ElapsedMilliseconds}ms");

                // 调试：打印前几个像素值
                Debug.WriteLine($"[ML Test] First 10 R values: {string.Join(", ", _cachedRgbData.Take(10).Select(v => v.ToString("F3")))}");
                Debug.WriteLine($"[ML Test] First 10 G values: {string.Join(", ", _cachedRgbData.Skip(targetSize * targetSize).Take(10).Select(v => v.ToString("F3")))}");
                Debug.WriteLine($"[ML Test] First 10 B values: {string.Join(", ", _cachedRgbData.Skip(2 * targetSize * targetSize).Take(10).Select(v => v.ToString("F3")))}");

                // 为了显示，转为 PNG（无损）
                sw.Restart();
                using var outputStream = new MemoryStream();
                scaledBitmap.Compress(Android.Graphics.Bitmap.CompressFormat.Png!, 100, outputStream);
                Debug.WriteLine($"[ML Test] PNG for display: {outputStream.Length / 1024.0:F1} KB, took {sw.ElapsedMilliseconds}ms");

                return outputStream.ToArray();
            });
#else
            // 非 Android 平台，直接使用原始数据
            _currentImageBytes = await Task.Run(() =>
            {
                originalStream.Position = 0;
                return originalStream.ToArray();
            });
#endif
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[ML Test] Image processing failed: {ex.Message}, using original");
            originalStream.Position = 0;
            _currentImageBytes = originalStream.ToArray();
            _cachedRgbData = null;
        }

        // 显示图片
        TestImage.Source = ImageSource.FromStream(() => new MemoryStream(_currentImageBytes));

        // 更新 UI
        OnPropertyChanged(nameof(HasImage));
        HasResult = false;
        OnPropertyChanged(nameof(HasResult));

        Debug.WriteLine($"[ML Test] Image ready: {_currentImageBytes.Length / 1024.0:F1} KB");
    }

    private async void OnDetectClicked(object sender, EventArgs e)
    {
        if (_currentImageBytes == null)
            return;

        try
        {
            DetectButton.IsEnabled = false;
            DetectButton.Text = "检测中...";

            // 记录开始时间
            var stopwatch = Stopwatch.StartNew();

            if (_originalImageBytes == null)
            {
                await DisplayAlert("错误", "图片数据未正确加载", "确定");
                return;
            }

#if ANDROID
            // Android: 用 Android Bitmap 缩放到 512x512，然后提取像素给 ML
            // 关键：确保像素提取方式与训练时一致
            Debug.WriteLine($"[ML Test] Preparing 512x512 image for ML (Android native)...");

            var (rgbData, preparedBytes) = await Task.Run(() =>
            {
                using var bitmap = Android.Graphics.BitmapFactory.DecodeByteArray(_originalImageBytes, 0, _originalImageBytes.Length);
                if (bitmap == null)
                    throw new Exception("无法解码图片");

                const int targetSize = 512;

                // 使用高质量缩放
                using var scaledBitmap = Android.Graphics.Bitmap.CreateScaledBitmap(bitmap, targetSize, targetSize, true);

                // 提取像素
                int[] pixels = new int[targetSize * targetSize];
                scaledBitmap.GetPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize);

                // 转换为 CHW 格式的 float 数组
                float[] rgb = new float[3 * targetSize * targetSize];
                for (int y = 0; y < targetSize; y++)
                {
                    for (int x = 0; x < targetSize; x++)
                    {
                        int pixel = pixels[y * targetSize + x];
                        int r = (pixel >> 16) & 0xFF;
                        int g = (pixel >> 8) & 0xFF;
                        int b = pixel & 0xFF;

                        int idx = y * targetSize + x;
                        rgb[idx] = r / 255f;
                        rgb[targetSize * targetSize + idx] = g / 255f;
                        rgb[2 * targetSize * targetSize + idx] = b / 255f;
                    }
                }

                // 同时保存为 PNG 用于对比调试
                using var ms = new MemoryStream();
                scaledBitmap.Compress(Android.Graphics.Bitmap.CompressFormat.Png!, 100, ms);

                Debug.WriteLine($"[ML Test] RGB data ready: {rgb.Length} floats");
                Debug.WriteLine($"[ML Test] RGB stats - min: {rgb.Min():F3}, max: {rgb.Max():F3}, mean: {rgb.Average():F3}");
                Debug.WriteLine($"[ML Test] First 10 R: {string.Join(", ", rgb.Take(10).Select(v => v.ToString("F3")))}");
                Debug.WriteLine($"[ML Test] First 10 G: {string.Join(", ", rgb.Skip(targetSize * targetSize).Take(10).Select(v => v.ToString("F3")))}");
                Debug.WriteLine($"[ML Test] First 10 B: {string.Join(", ", rgb.Skip(2 * targetSize * targetSize).Take(10).Select(v => v.ToString("F3")))}");

                return (rgb, ms.ToArray());
            });

            Debug.WriteLine($"[ML Test] Running ML inference...");
            var result = await _mlService.DetectCornersFromRgbAsync(rgbData, _originalWidth, _originalHeight, _originalImageBytes);
#else
            Debug.WriteLine($"[ML Test] Using original image bytes: {_originalImageBytes.Length} bytes");
            var result = await _mlService.DetectCornersAsync(_originalImageBytes);
#endif

            stopwatch.Stop();

            // 显示推理时间
            InferenceTimeLabel.Text = $"推理耗时: {stopwatch.ElapsedMilliseconds} ms";

            // 显示 ML 原始输出
            var mlCorners = result.MLRawCorners ?? result.Corners;
            MLCornersLabel.Text = $"TL: ({mlCorners.TopLeftX:F1}, {mlCorners.TopLeftY:F1})\n" +
                $"TR: ({mlCorners.TopRightX:F1}, {mlCorners.TopRightY:F1})\n" +
                $"BR: ({mlCorners.BottomRightX:F1}, {mlCorners.BottomRightY:F1})\n" +
                $"BL: ({mlCorners.BottomLeftX:F1}, {mlCorners.BottomLeftY:F1})";

            // 显示归一化坐标
            if (result.NormalizedCoordinates != null)
            {
                var nc = result.NormalizedCoordinates;
                MLNormalizedLabel.Text = $"归一化:\n[{nc[0]:F3},{nc[1]:F3}]\n[{nc[2]:F3},{nc[3]:F3}]\n[{nc[4]:F3},{nc[5]:F3}]\n[{nc[6]:F3},{nc[7]:F3}]";
            }

            // 显示 CV 精修结果
            var refinedCorners = result.Corners;
            RefinedCornersLabel.Text = $"TL: ({refinedCorners.TopLeftX:F1}, {refinedCorners.TopLeftY:F1})\n" +
                $"TR: ({refinedCorners.TopRightX:F1}, {refinedCorners.TopRightY:F1})\n" +
                $"BR: ({refinedCorners.BottomRightX:F1}, {refinedCorners.BottomRightY:F1})\n" +
                $"BL: ({refinedCorners.BottomLeftX:F1}, {refinedCorners.BottomLeftY:F1})";

            // 计算 ML 和 CV 精修的差异
            float diffTL = Distance(mlCorners.TopLeftX, mlCorners.TopLeftY, refinedCorners.TopLeftX, refinedCorners.TopLeftY);
            float diffTR = Distance(mlCorners.TopRightX, mlCorners.TopRightY, refinedCorners.TopRightX, refinedCorners.TopRightY);
            float diffBR = Distance(mlCorners.BottomRightX, mlCorners.BottomRightY, refinedCorners.BottomRightX, refinedCorners.BottomRightY);
            float diffBL = Distance(mlCorners.BottomLeftX, mlCorners.BottomLeftY, refinedCorners.BottomLeftX, refinedCorners.BottomLeftY);
            float avgDiff = (diffTL + diffTR + diffBR + diffBL) / 4;
            RefinementDiffLabel.Text = $"差异:\nTL={diffTL:F1}px TR={diffTR:F1}px\nBR={diffBR:F1}px BL={diffBL:F1}px\n平均={avgDiff:F1}px";

            // 如果是 test.jpg（4080x3060），计算 MSE Loss（和训练时一样）
            if (TestGroundTruth != null && _originalWidth == TestImageWidth && _originalHeight == TestImageHeight)
            {
                // 将真实坐标归一化到 [0,1]
                float[] gtNormalized = new float[8]
                {
                    TestGroundTruth.TopLeftX / TestImageWidth,
                    TestGroundTruth.TopLeftY / TestImageHeight,
                    TestGroundTruth.TopRightX / TestImageWidth,
                    TestGroundTruth.TopRightY / TestImageHeight,
                    TestGroundTruth.BottomRightX / TestImageWidth,
                    TestGroundTruth.BottomRightY / TestImageHeight,
                    TestGroundTruth.BottomLeftX / TestImageWidth,
                    TestGroundTruth.BottomLeftY / TestImageHeight
                };

                // 计算 MSE Loss（和训练脚本完全一样）
                float mseLoss = 0f;
                for (int i = 0; i < 8; i++)
                {
                    float diff = result.NormalizedCoordinates![i] - gtNormalized[i];
                    mseLoss += diff * diff;
                }
                mseLoss /= 8;  // 平均

                // 计算像素误差（用于直观理解）
                float gtDiffTL = Distance(mlCorners.TopLeftX, mlCorners.TopLeftY, TestGroundTruth.TopLeftX, TestGroundTruth.TopLeftY);
                float gtDiffTR = Distance(mlCorners.TopRightX, mlCorners.TopRightY, TestGroundTruth.TopRightX, TestGroundTruth.TopRightY);
                float gtDiffBR = Distance(mlCorners.BottomRightX, mlCorners.BottomRightY, TestGroundTruth.BottomRightX, TestGroundTruth.BottomRightY);
                float gtDiffBL = Distance(mlCorners.BottomLeftX, mlCorners.BottomLeftY, TestGroundTruth.BottomLeftX, TestGroundTruth.BottomLeftY);
                float gtAvgDiff = (gtDiffTL + gtDiffTR + gtDiffBR + gtDiffBL) / 4;

                RefinementDiffLabel.Text += $"\n\n【test.jpg Loss】\nMSE Loss: {mseLoss:F6}\n平均像素误差: {gtAvgDiff:F1}px";

                Debug.WriteLine($"[ML Test] test.jpg MSE Loss: {mseLoss:F6}, avg pixel error: {gtAvgDiff:F1}px");
            }

            HasResult = true;
            OnPropertyChanged(nameof(HasResult));

            Debug.WriteLine($"[ML Test] Detection completed in {stopwatch.ElapsedMilliseconds}ms");
            Debug.WriteLine($"[ML Test] ML corners: TL({mlCorners.TopLeftX:F1},{mlCorners.TopLeftY:F1}) TR({mlCorners.TopRightX:F1},{mlCorners.TopRightY:F1})");
            Debug.WriteLine($"[ML Test] Refined corners: TL({refinedCorners.TopLeftX:F1},{refinedCorners.TopLeftY:F1}) TR({refinedCorners.TopRightX:F1},{refinedCorners.TopRightY:F1})");
            Debug.WriteLine($"[ML Test] Refinement diff: avg={avgDiff:F1}px");

            // 绘制可视化对比图（ML原始 vs CV精修）
            await DrawVisualizationAsync(mlCorners, refinedCorners);

            // 执行透视变换并显示结果
            await PerformPerspectiveTransformAsync(result.Corners);
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"检测失败: {ex.Message}", "确定");
            Debug.WriteLine($"[ML Test] Error: {ex}");
        }
        finally
        {
            DetectButton.IsEnabled = true;
            DetectButton.Text = "🔍 开始检测";
        }
    }

    private async void OnLoadModelInfoClicked(object sender, EventArgs e)
    {
        try
        {
            var modelInfo = await _mlService.GetModelInfoAsync();

            ModelInfoLabel.Text = $"版本: {modelInfo.Version}\n" +
                $"文件大小: {modelInfo.FileSizeBytes / (1024.0 * 1024.0):F2} MB\n" +
                $"训练日期: {modelInfo.TrainedDate:yyyy-MM-dd}\n" +
                $"描述: {modelInfo.Description}";

            ModelInfoLabel.IsVisible = true;
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"无法加载模型信息: {ex.Message}", "确定");
        }
    }

    private async void OnCopyErrorClicked(object sender, EventArgs e)
    {
        if (!string.IsNullOrEmpty(_lastErrorMessage))
        {
            await Clipboard.SetTextAsync(_lastErrorMessage);
            await DisplayAlert("已复制", "错误消息已复制到剪贴板", "确定");
        }
    }

    private async Task PerformPerspectiveTransformAsync(QuadrilateralPoints corners)
    {
        if (_originalImageBytes == null)
            return;

        try
        {
            Debug.WriteLine($"[ML Test] Starting perspective transform...");

            Func<byte[]?> transformFunc = () =>
            {
#if ANDROID
                // 加载原始图片
                using var bitmap = Android.Graphics.BitmapFactory.DecodeByteArray(_originalImageBytes, 0, _originalImageBytes.Length);
                if (bitmap == null)
                    return null;

                var width = bitmap.Width;
                var height = bitmap.Height;

                Debug.WriteLine($"[ML Test] Transform source: {width}x{height}");

                // 源点（检测到的四个角点）
                float[] src = new float[] {
                    corners.TopLeftX, corners.TopLeftY,
                    corners.TopRightX, corners.TopRightY,
                    corners.BottomRightX, corners.BottomRightY,
                    corners.BottomLeftX, corners.BottomLeftY
                };

                // 计算目标图片尺寸（保持宽高比）
                float srcWidth = Math.Max(
                    Distance(corners.TopLeftX, corners.TopLeftY, corners.TopRightX, corners.TopRightY),
                    Distance(corners.BottomLeftX, corners.BottomLeftY, corners.BottomRightX, corners.BottomRightY)
                );
                float srcHeight = Math.Max(
                    Distance(corners.TopLeftX, corners.TopLeftY, corners.BottomLeftX, corners.BottomLeftY),
                    Distance(corners.TopRightX, corners.TopRightY, corners.BottomRightX, corners.BottomRightY)
                );

                int dstWidth = (int)srcWidth;
                int dstHeight = (int)srcHeight;

                Debug.WriteLine($"[ML Test] Transform target: {dstWidth}x{dstHeight}");

                // 目标点（矩形）
                float[] dst = new float[] {
                    0, 0,
                    dstWidth, 0,
                    dstWidth, dstHeight,
                    0, dstHeight
                };

                // 计算透视变换矩阵
                var matrix = new Android.Graphics.Matrix();
                matrix.SetPolyToPoly(src, 0, dst, 0, 4);

                // 创建变换后的 Bitmap
                using var transformedBitmap = Android.Graphics.Bitmap.CreateBitmap(dstWidth, dstHeight, Android.Graphics.Bitmap.Config.Argb8888!);
                using var canvas = new Android.Graphics.Canvas(transformedBitmap);
                canvas.DrawBitmap(bitmap, matrix, new Android.Graphics.Paint { FilterBitmap = true });

                // 转换为 JPEG 字节
                using var outputStream = new MemoryStream();
                transformedBitmap.Compress(Android.Graphics.Bitmap.CompressFormat.Jpeg!, 90, outputStream);

                Debug.WriteLine($"[ML Test] Transform completed: {outputStream.Length / 1024.0:F1} KB");
                return outputStream.ToArray();
#else
                return null;
#endif
            };

            byte[]? transformedBytes = await Task.Run(transformFunc);

            if (transformedBytes != null)
            {
                // 显示变换后的图片
                TransformedImage.Source = ImageSource.FromStream(() => new MemoryStream(transformedBytes));
                Debug.WriteLine($"[ML Test] Transformed image displayed");
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[ML Test] Perspective transform failed: {ex.Message}");
        }
    }

    /// <summary>
    /// 在原图上绘制 ML 原始输出和 CV 精修结果的可视化对比
    /// </summary>
    private async Task DrawVisualizationAsync(QuadrilateralPoints mlCorners, QuadrilateralPoints refinedCorners)
    {
        if (_originalImageBytes == null)
            return;

        try
        {
            Debug.WriteLine($"[ML Test] Drawing visualization...");

            Func<byte[]?> createVisualization = () =>
            {
#if ANDROID
                // 加载原始图片
                using var bitmap = Android.Graphics.BitmapFactory.DecodeByteArray(_originalImageBytes, 0, _originalImageBytes.Length);
                if (bitmap == null)
                    return null;

                // 创建可变副本用于绘制
                using var mutableBitmap = bitmap.Copy(Android.Graphics.Bitmap.Config.Argb8888!, true);
                if (mutableBitmap == null)
                    return null;

                using var canvas = new Android.Graphics.Canvas(mutableBitmap);

                // 计算画笔大小（基于图片尺寸）
                float strokeWidth = Math.Max(4f, Math.Min(bitmap.Width, bitmap.Height) / 200f);
                float radius = strokeWidth * 3;

                // ML 原始输出：蓝色
                using var mlPaint = new Android.Graphics.Paint
                {
                    AntiAlias = true,
                    StrokeWidth = strokeWidth
                };
                mlPaint.SetStyle(Android.Graphics.Paint.Style.Stroke);
                mlPaint.Color = Android.Graphics.Color.Blue;

                var mlPoints = new[]
                {
                    (mlCorners.TopLeftX, mlCorners.TopLeftY),
                    (mlCorners.TopRightX, mlCorners.TopRightY),
                    (mlCorners.BottomRightX, mlCorners.BottomRightY),
                    (mlCorners.BottomLeftX, mlCorners.BottomLeftY)
                };

                // 画 ML 四边形
                for (int i = 0; i < 4; i++)
                {
                    var p1 = mlPoints[i];
                    var p2 = mlPoints[(i + 1) % 4];
                    canvas.DrawLine(p1.Item1, p1.Item2, p2.Item1, p2.Item2, mlPaint);
                }

                // 画 ML 角点
                foreach (var pt in mlPoints)
                {
                    canvas.DrawCircle(pt.Item1, pt.Item2, radius, mlPaint);
                }

                // CV 精修结果：红色
                using var refinedPaint = new Android.Graphics.Paint
                {
                    AntiAlias = true,
                    StrokeWidth = strokeWidth
                };
                refinedPaint.SetStyle(Android.Graphics.Paint.Style.Stroke);
                refinedPaint.Color = Android.Graphics.Color.Red;

                var refinedPoints = new[]
                {
                    (refinedCorners.TopLeftX, refinedCorners.TopLeftY),
                    (refinedCorners.TopRightX, refinedCorners.TopRightY),
                    (refinedCorners.BottomRightX, refinedCorners.BottomRightY),
                    (refinedCorners.BottomLeftX, refinedCorners.BottomLeftY)
                };

                // 画精修后四边形
                for (int i = 0; i < 4; i++)
                {
                    var p1 = refinedPoints[i];
                    var p2 = refinedPoints[(i + 1) % 4];
                    canvas.DrawLine(p1.Item1, p1.Item2, p2.Item1, p2.Item2, refinedPaint);
                }

                // 画精修后角点
                foreach (var pt in refinedPoints)
                {
                    canvas.DrawCircle(pt.Item1, pt.Item2, radius, refinedPaint);
                }

                // 差异连线：绿色
                using var diffPaint = new Android.Graphics.Paint
                {
                    AntiAlias = true,
                    StrokeWidth = strokeWidth / 2
                };
                diffPaint.SetStyle(Android.Graphics.Paint.Style.Stroke);
                diffPaint.Color = Android.Graphics.Color.Lime;

                for (int i = 0; i < 4; i++)
                {
                    var ml = mlPoints[i];
                    var refined = refinedPoints[i];
                    canvas.DrawLine(ml.Item1, ml.Item2, refined.Item1, refined.Item2, diffPaint);
                }

                // 转换为 JPEG 字节
                using var outputStream = new MemoryStream();
                mutableBitmap.Compress(Android.Graphics.Bitmap.CompressFormat.Jpeg!, 90, outputStream);

                Debug.WriteLine($"[ML Test] Visualization created: {outputStream.Length / 1024.0:F1} KB");
                return outputStream.ToArray();
#else
                return null;
#endif
            };

            byte[]? visualizationBytes = await Task.Run(createVisualization);

            if (visualizationBytes != null)
            {
                VisualizationImage.Source = ImageSource.FromStream(() => new MemoryStream(visualizationBytes));
                Debug.WriteLine($"[ML Test] Visualization displayed");
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[ML Test] Visualization failed: {ex.Message}");
        }
    }

    private static float Distance(float x1, float y1, float x2, float y2)
    {
        float dx = x2 - x1;
        float dy = y2 - y1;
        return (float)Math.Sqrt(dx * dx + dy * dy);
    }
}
