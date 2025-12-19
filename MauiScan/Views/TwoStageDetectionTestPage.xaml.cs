using MauiScan.Models;
using MauiScan.Services;
using System.Diagnostics;

namespace MauiScan.Views;

public partial class TwoStageDetectionTestPage : ContentPage
{
    private readonly TwoStageDetectionService _detectionService;
    private TwoStageDetectionResult? _lastResult;
    private byte[]? _currentImageBytes;

    public TwoStageDetectionTestPage(TwoStageDetectionService detectionService)
    {
        InitializeComponent();
        _detectionService = detectionService;
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
                    await LoadAndProcessImageAsync(photo);
                    Debug.WriteLine($"[TwoStageTest] Photo captured: {photo.FileName}");
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
            Debug.WriteLine($"[TwoStageTest] Camera error: {ex}");
        }
    }

    private async void OnPickImageClicked(object sender, EventArgs e)
    {
        try
        {
            var result = await FilePicker.PickAsync(new PickOptions
            {
                PickerTitle = "选择幕布+PPT照片",
                FileTypes = FilePickerFileType.Images
            });

            if (result != null)
            {
                await LoadAndProcessImageAsync(result);
                Debug.WriteLine($"[TwoStageTest] Image loaded: {result.FileName}");
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"无法加载图片: {ex.Message}", "确定");
            Debug.WriteLine($"[TwoStageTest] File picker error: {ex}");
        }
    }

    private void OnClearClicked(object sender, EventArgs e)
    {
        ResultImage.Source = null;
        _lastResult = null;
        _currentImageBytes = null;

        ScreenStatusLabel.Text = "⏸️ 未检测";
        ScreenConfidenceLabel.Text = "置信度: -";
        ScreenQuadLabel.Text = "";

        PptStatusLabel.Text = "⏸️ 未检测";
        PptConfidenceLabel.Text = "置信度: -";
        PptQuadLabel.Text = "";

        SummaryBorder.IsVisible = false;
        StatusLabel.Text = "已清除 - 选择新图片开始测试";
    }

    private async Task LoadAndProcessImageAsync(FileResult fileResult)
    {
        try
        {
            Debug.WriteLine($"[TwoStageTest] Loading image: {fileResult.FileName}");
            StatusLabel.Text = "正在加载图片...";

            // 读取图片
            using var stream = await fileResult.OpenReadAsync();
            using var memoryStream = new MemoryStream();
            await stream.CopyToAsync(memoryStream);

            _currentImageBytes = memoryStream.ToArray();

            Debug.WriteLine($"[TwoStageTest] Image loaded: {_currentImageBytes.Length / 1024.0:F1} KB");

            // 开始两阶段检测
            await ProcessImageAsync(_currentImageBytes);
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"处理图片失败: {ex.Message}", "确定");
            Debug.WriteLine($"[TwoStageTest] Process error: {ex}");
            StatusLabel.Text = $"错误: {ex.Message}";
        }
    }

    private async Task ProcessImageAsync(byte[] imageBytes)
    {
        try
        {
            StatusLabel.Text = "🔍 正在执行两阶段检测...";

            var stopwatch = Stopwatch.StartNew();

            // 执行两阶段检测
            _lastResult = await _detectionService.DetectAsync(imageBytes);

            stopwatch.Stop();

            Debug.WriteLine($"[TwoStageTest] Detection completed in {stopwatch.ElapsedMilliseconds}ms");

            // 绘制可视化结果
            var visualizedImage = DrawDetectionBoxes(_lastResult);
            if (visualizedImage != null)
            {
                ResultImage.Source = ImageSource.FromStream(() => new MemoryStream(visualizedImage));
            }
            else
            {
                // 如果绘制失败，显示原图
                ResultImage.Source = ImageSource.FromStream(() => new MemoryStream(imageBytes));
            }

            // 更新 UI
            UpdateScreenResult(_lastResult.ScreenStage);
            UpdatePptResult(_lastResult.PptStage);
            UpdateSummary(_lastResult, stopwatch.ElapsedMilliseconds);

            // 更新状态栏
            if (_lastResult.BothStagesSuccess)
            {
                StatusLabel.Text = $"✅ 两阶段检测成功 (耗时 {stopwatch.ElapsedMilliseconds}ms)";
            }
            else if (_lastResult.ScreenStage.IsSuccess)
            {
                StatusLabel.Text = $"⚠️ 仅幕布检测成功 (耗时 {stopwatch.ElapsedMilliseconds}ms)";
            }
            else
            {
                StatusLabel.Text = $"❌ 检测失败 (耗时 {stopwatch.ElapsedMilliseconds}ms)";
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("检测失败", ex.Message, "确定");
            Debug.WriteLine($"[TwoStageTest] Detection error: {ex}");
            StatusLabel.Text = $"❌ 检测失败: {ex.Message}";
        }
    }

    private byte[]? DrawDetectionBoxes(TwoStageDetectionResult result)
    {
        try
        {
#if ANDROID
            Debug.WriteLine("[TwoStageTest] Drawing detection boxes...");

            // 加载原始图片
            using var bitmap = Android.Graphics.BitmapFactory.DecodeByteArray(
                result.OriginalImageBytes, 0, result.OriginalImageBytes.Length);

            if (bitmap == null)
            {
                Debug.WriteLine("[TwoStageTest] Failed to decode bitmap");
                return null;
            }

            // 创建可变副本
            using var mutableBitmap = bitmap.Copy(Android.Graphics.Bitmap.Config.Argb8888!, true);
            if (mutableBitmap == null)
            {
                Debug.WriteLine("[TwoStageTest] Failed to create mutable bitmap");
                return null;
            }

            using var canvas = new Android.Graphics.Canvas(mutableBitmap);

            // 计算线条粗细（基于图片尺寸）
            float strokeWidth = Math.Max(5f, Math.Min(bitmap.Width, bitmap.Height) / 150f);
            float radius = strokeWidth * 2.5f;

            Debug.WriteLine($"[TwoStageTest] Stroke width: {strokeWidth}, radius: {radius}");

            // 绘制幕布检测框（绿色）
            if (result.ScreenStage.IsSuccess && result.ScreenStage.Quad != null)
            {
                DrawQuadrilateral(canvas, result.ScreenStage.Quad, Android.Graphics.Color.Green, strokeWidth, radius);
                Debug.WriteLine("[TwoStageTest] Screen quad drawn (green)");
            }

            // 绘制PPT检测框（红色）
            if (result.PptStage?.IsSuccess == true && result.PptStage.Quad != null)
            {
                DrawQuadrilateral(canvas, result.PptStage.Quad, Android.Graphics.Color.Red, strokeWidth, radius);
                Debug.WriteLine("[TwoStageTest] PPT quad drawn (red)");
            }

            // 转换为 JPEG
            using var outputStream = new MemoryStream();
            mutableBitmap.Compress(Android.Graphics.Bitmap.CompressFormat.Jpeg!, 90, outputStream);

            Debug.WriteLine($"[TwoStageTest] Visualization created: {outputStream.Length / 1024.0:F1} KB");

            return outputStream.ToArray();
#else
            Debug.WriteLine("[TwoStageTest] Drawing not supported on non-Android platforms");
            return null;
#endif
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[TwoStageTest] Drawing error: {ex.Message}");
            return null;
        }
    }

#if ANDROID
    private void DrawQuadrilateral(
        Android.Graphics.Canvas canvas,
        QuadrilateralPoints quad,
        Android.Graphics.Color color,
        float strokeWidth,
        float radius)
    {
        using var paint = new Android.Graphics.Paint
        {
            AntiAlias = true,
            StrokeWidth = strokeWidth,
            Color = color
        };

        paint.SetStyle(Android.Graphics.Paint.Style.Stroke);

        var points = new[]
        {
            (quad.TopLeft.X, quad.TopLeft.Y),
            (quad.TopRight.X, quad.TopRight.Y),
            (quad.BottomRight.X, quad.BottomRight.Y),
            (quad.BottomLeft.X, quad.BottomLeft.Y)
        };

        // 画四条边
        for (int i = 0; i < 4; i++)
        {
            var p1 = points[i];
            var p2 = points[(i + 1) % 4];
            canvas.DrawLine(p1.Item1, p1.Item2, p2.Item1, p2.Item2, paint);
        }

        // 画四个角点
        paint.SetStyle(Android.Graphics.Paint.Style.Fill);
        foreach (var pt in points)
        {
            canvas.DrawCircle(pt.Item1, pt.Item2, radius, paint);
        }
    }
#endif

    private void UpdateScreenResult(StageResult stage)
    {
        ScreenStatusLabel.Text = stage.IsSuccess ? "✅ 成功" : $"❌ {stage.ErrorMessage}";
        ScreenConfidenceLabel.Text = stage.IsSuccess ? $"置信度: {stage.Confidence:P0}" : "置信度: -";

        if (stage.Quad != null)
        {
            ScreenQuadLabel.Text =
                $"TL: ({stage.Quad.TopLeft.X}, {stage.Quad.TopLeft.Y})\n" +
                $"TR: ({stage.Quad.TopRight.X}, {stage.Quad.TopRight.Y})\n" +
                $"BR: ({stage.Quad.BottomRight.X}, {stage.Quad.BottomRight.Y})\n" +
                $"BL: ({stage.Quad.BottomLeft.X}, {stage.Quad.BottomLeft.Y})";
        }
        else
        {
            ScreenQuadLabel.Text = stage.IsSuccess ? "" : stage.ErrorMessage;
        }
    }

    private void UpdatePptResult(StageResult? stage)
    {
        if (stage == null)
        {
            PptStatusLabel.Text = "⏭️ 未执行（幕布检测失败）";
            PptConfidenceLabel.Text = "置信度: -";
            PptQuadLabel.Text = "";
            return;
        }

        PptStatusLabel.Text = stage.IsSuccess ? "✅ 成功" : $"❌ {stage.ErrorMessage}";
        PptConfidenceLabel.Text = stage.IsSuccess ? $"置信度: {stage.Confidence:P0}" : "置信度: -";

        if (stage.Quad != null)
        {
            PptQuadLabel.Text =
                $"TL: ({stage.Quad.TopLeft.X}, {stage.Quad.TopLeft.Y})\n" +
                $"TR: ({stage.Quad.TopRight.X}, {stage.Quad.TopRight.Y})\n" +
                $"BR: ({stage.Quad.BottomRight.X}, {stage.Quad.BottomRight.Y})\n" +
                $"BL: ({stage.Quad.BottomLeft.X}, {stage.Quad.BottomLeft.Y})";
        }
        else
        {
            PptQuadLabel.Text = stage.IsSuccess ? "" : stage.ErrorMessage;
        }
    }

    private void UpdateSummary(TwoStageDetectionResult result, long elapsedMs)
    {
        SummaryBorder.IsVisible = true;

        string summary = $"图像尺寸: {result.OriginalSize.Width} × {result.OriginalSize.Height}\n" +
                        $"检测耗时: {elapsedMs} ms\n" +
                        $"幕布检测: {(result.ScreenStage.IsSuccess ? "成功" : "失败")}\n" +
                        $"PPT检测: {(result.PptStage?.IsSuccess == true ? "成功" : "失败")}\n";

        if (result.BothStagesSuccess)
        {
            summary += "✅ 两阶段检测均成功，可用于裁剪PPT内容";
        }
        else if (result.ScreenStage.IsSuccess)
        {
            summary += "⚠️ 仅幕布检测成功，可用于裁剪幕布内容";
        }
        else
        {
            summary += "❌ 检测失败，无法进行裁剪";
        }

        SummaryLabel.Text = summary;
    }
}
