using MauiScan.ML.Services;
using MauiScan.ML.Models;
using System.Diagnostics;
#if !ANDROID && !IOS && !MACCATALYST
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
#endif

namespace MauiScan.Views;

public partial class MLTestPage : ContentPage
{
    private readonly IMLInferenceService _mlService;
    private byte[]? _currentImageBytes;      // 512x512 缩放后的图片（用于推理）
    private byte[]? _originalImageBytes;     // 原始图片（用于透视变换）
    private int _originalWidth;              // 原始图片宽度
    private int _originalHeight;             // 原始图片高度
    private string? _lastErrorMessage;

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
            Debug.WriteLine($"[ML Test] OnAppearing started");
            Debug.WriteLine($"[ML Test] AppDataDirectory: {FileSystem.AppDataDirectory}");

            await App.EnsureModelFileCopiedAsync();
            Debug.WriteLine($"[ML Test] Model file copy completed");

            await CheckModelStatusAsync();
            Debug.WriteLine($"[ML Test] Model status check completed");

            HasError = false;
            OnPropertyChanged(nameof(HasError));
        }
        catch (Exception ex)
        {
            var innerMsg = ex.InnerException != null ? $"\n\n内部异常: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}" : "";
            _lastErrorMessage = $"初始化失败\n\n错误类型: {ex.GetType().Name}\n错误消息: {ex.Message}{innerMsg}\n\n完整堆栈:\n{ex.StackTrace}";
            ModelStatusLabel.Text = $"❌ 初始化失败: {ex.GetType().Name}\n{ex.Message}";
            ModelStatusLabel.TextColor = Colors.Red;
            Debug.WriteLine($"[ML Test] Error during initialization: {ex}");
            if (ex.InnerException != null)
            {
                Debug.WriteLine($"[ML Test] Inner exception: {ex.InnerException}");
            }

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
        // 避免在推理服务中使用 ImageSharp（在 Android 上极慢）
        const int targetSize = 512;

        try
        {
            // 在后台线程处理图片缩放
            _currentImageBytes = await Task.Run(async () =>
            {
#if ANDROID
                using var bitmap = Android.Graphics.BitmapFactory.DecodeStream(originalStream);
                if (bitmap == null)
                    throw new Exception("无法解码图片");

                _originalWidth = bitmap.Width;
                _originalHeight = bitmap.Height;
                Debug.WriteLine($"[ML Test] Original dimensions: {bitmap.Width}x{bitmap.Height}");

                // 强制缩放到 512x512（拉伸，不保持宽高比）
                // 这与训练时的预处理一致
                Debug.WriteLine($"[ML Test] Resizing to: {targetSize}x{targetSize}");

                var sw = System.Diagnostics.Stopwatch.StartNew();
                using var scaledBitmap = Android.Graphics.Bitmap.CreateScaledBitmap(bitmap, targetSize, targetSize, true);
                Debug.WriteLine($"[ML Test] Android resize took: {sw.ElapsedMilliseconds}ms");

                using var outputStream = new MemoryStream();
                await scaledBitmap.CompressAsync(Android.Graphics.Bitmap.CompressFormat.Jpeg!, 95, outputStream);

                Debug.WriteLine($"[ML Test] Final image size: {outputStream.Length / 1024.0:F1} KB");
                return outputStream.ToArray();
#elif IOS || MACCATALYST
                // iOS/MacCatalyst 平台：使用 UIKit 处理图片
                originalStream.Position = 0;
                using var uiImage = UIKit.UIImage.LoadFromData(Foundation.NSData.FromArray(originalStream.ToArray()));
                if (uiImage == null)
                    throw new Exception("无法解码图片");

                _originalWidth = (int)uiImage.Size.Width;
                _originalHeight = (int)uiImage.Size.Height;
                Debug.WriteLine($"[ML Test] Original dimensions: {_originalWidth}x{_originalHeight}");

                // 缩放到 512x512
                Debug.WriteLine($"[ML Test] Resizing to: {targetSize}x{targetSize}");
                var sw = System.Diagnostics.Stopwatch.StartNew();

                UIKit.UIGraphics.BeginImageContextWithOptions(new CoreGraphics.CGSize(targetSize, targetSize), false, 1.0f);
                uiImage.Draw(new CoreGraphics.CGRect(0, 0, targetSize, targetSize));
                var scaledImage = UIKit.UIGraphics.GetImageFromCurrentImageContext();
                UIKit.UIGraphics.EndImageContext();

                Debug.WriteLine($"[ML Test] iOS resize took: {sw.ElapsedMilliseconds}ms");

                // 转换为 JPEG
                using var jpegData = scaledImage?.AsJPEG(0.95f);
                if (jpegData == null)
                    throw new Exception("无法压缩图片");

                var bytes = new byte[jpegData.Length];
                System.Runtime.InteropServices.Marshal.Copy(jpegData.Bytes, bytes, 0, (int)jpegData.Length);

                Debug.WriteLine($"[ML Test] Final image size: {bytes.Length / 1024.0:F1} KB");
                return bytes;
#else
                // 其他平台，使用 ImageSharp
                originalStream.Position = 0;
                using var image = Image.Load<SixLabors.ImageSharp.PixelFormats.Rgb24>(originalStream);
                _originalWidth = image.Width;
                _originalHeight = image.Height;
                Debug.WriteLine($"[ML Test] Original dimensions: {_originalWidth}x{_originalHeight}");

                image.Mutate(x => x.Resize(targetSize, targetSize));
                using var outputStream = new MemoryStream();
                image.SaveAsJpeg(outputStream);
                return outputStream.ToArray();
#endif
            });
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[ML Test] Image resize failed: {ex.Message}, using original");
            originalStream.Position = 0;
            _currentImageBytes = originalStream.ToArray();
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

#if ANDROID
            // Android 平台：使用原生 API 提取 RGB 数据
            float[]? rgbData = await Task.Run(() =>
            {
                using var bitmap = Android.Graphics.BitmapFactory.DecodeByteArray(_currentImageBytes, 0, _currentImageBytes.Length!);
                if (bitmap == null)
                    return null;

                Debug.WriteLine($"[ML Test] Extracting RGB from {bitmap.Width}x{bitmap.Height} bitmap");

                // 提取像素
                int[] pixels = new int[bitmap.Width * bitmap.Height];
                bitmap.GetPixels(pixels, 0, bitmap.Width, 0, 0, bitmap.Width, bitmap.Height);

                // 转换为 CHW 格式的 float 数组
                float[] rgb = new float[3 * bitmap.Width * bitmap.Height];
                for (int y = 0; y < bitmap.Height; y++)
                {
                    for (int x = 0; x < bitmap.Width; x++)
                    {
                        int pixel = pixels[y * bitmap.Width + x];
                        int r = (pixel >> 16) & 0xFF;
                        int g = (pixel >> 8) & 0xFF;
                        int b = pixel & 0xFF;

                        int idx = y * bitmap.Width + x;
                        rgb[idx] = r / 255f;                                      // R 通道
                        rgb[bitmap.Width * bitmap.Height + idx] = g / 255f;      // G 通道
                        rgb[2 * bitmap.Width * bitmap.Height + idx] = b / 255f;  // B 通道
                    }
                }

                Debug.WriteLine($"[ML Test] RGB data extracted: {rgb.Length} floats");
                return rgb;
            });

            if (rgbData == null)
            {
                await DisplayAlert("错误", "无法提取图片数据", "确定");
                return;
            }

            // 运行 ML 推理（使用 RGB 数据）
            var result = await _mlService.DetectCornersFromRgbAsync(rgbData, _originalWidth, _originalHeight);
#else
            // 其他平台：使用 ImageSharp（慢）
            var result = await _mlService.DetectCornersAsync(_currentImageBytes, _originalWidth, _originalHeight);
#endif

            stopwatch.Stop();

            // 显示结果
            ConfidenceLabel.Text = $"置信度: {result.Confidence:P1}";

            string quality;
            Color qualityColor;
            if (result.IsHighQuality)
            {
                quality = "高质量 (直接使用 ML 结果)";
                qualityColor = Colors.Green;
            }
            else if (result.IsMediumQuality)
            {
                quality = "中等质量 (建议与传统算法融合)";
                qualityColor = Colors.Orange;
            }
            else
            {
                quality = "低质量 (降级使用传统算法)";
                qualityColor = Colors.Red;
            }

            QualityLabel.Text = $"质量评估: {quality}";
            QualityLabel.TextColor = qualityColor;

            var corners = result.Corners;
            CornersLabel.Text = $"检测到的角点:\n" +
                $"  左上: ({corners.TopLeftX:F1}, {corners.TopLeftY:F1})\n" +
                $"  右上: ({corners.TopRightX:F1}, {corners.TopRightY:F1})\n" +
                $"  右下: ({corners.BottomRightX:F1}, {corners.BottomRightY:F1})\n" +
                $"  左下: ({corners.BottomLeftX:F1}, {corners.BottomLeftY:F1})";

            InferenceTimeLabel.Text = $"推理耗时: {stopwatch.ElapsedMilliseconds} ms";

            HasResult = true;
            OnPropertyChanged(nameof(HasResult));

            Debug.WriteLine($"[ML Test] Detection completed in {stopwatch.ElapsedMilliseconds}ms");
            Debug.WriteLine($"[ML Test] Confidence: {result.Confidence:F3}");

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

    private static float Distance(float x1, float y1, float x2, float y2)
    {
        float dx = x2 - x1;
        float dy = y2 - y1;
        return (float)Math.Sqrt(dx * dx + dy * dy);
    }
}
