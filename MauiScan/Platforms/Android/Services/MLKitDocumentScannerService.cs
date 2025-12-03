using Android.App;
using Android.Gms.Extensions;
using Xamarin.Google.MLKit.Vision.DocumentScanner;
using MauiScan.Services;
using MauiScan.Models;
using AndroidContent = Android.Content;
using AndroidGraphics = Android.Graphics;

namespace MauiScan.Platforms.Android.Services;

/// <summary>
/// 基于 Google ML Kit 的文档扫描服务
/// </summary>
public class MLKitDocumentScannerService : IImageProcessingService
{
    private static TaskCompletionSource<GmsDocumentScanningResult?>? _scanCompletionSource;
    private static readonly int REQUEST_CODE_SCAN = 1001;

    public async Task<ScanResult> ProcessScanAsync(byte[] imageBytes, bool applyEnhancement = false)
    {
        // ML Kit 的 Document Scanner 是一个完整的扫描流程
        // 它自己处理相机、检测、裁剪，不需要我们传入图像
        // 这个方法保留是为了接口兼容，但实际扫描通过 StartScanAsync 进行
        return ScanResult.Failure("请使用 ML Kit 扫描功能");
    }

    public Task<QuadrilateralPoints?> DetectDocumentBoundsAsync(byte[] imageBytes)
    {
        // ML Kit Document Scanner 不支持单独的边界检测
        return Task.FromResult<QuadrilateralPoints?>(null);
    }

    public string GetVersion()
    {
        return "ML Kit Document Scanner 16.0.0";
    }

    /// <summary>
    /// 启动 ML Kit 文档扫描
    /// </summary>
    public static async Task<ScanResult> StartScanAsync(Activity activity)
    {
        try
        {
            System.Diagnostics.Debug.WriteLine("[MLKit] 开始初始化扫描器...");

            // 配置扫描选项
            var options = new GmsDocumentScannerOptions.Builder()
                .SetGalleryImportAllowed(true)      // 允许从相册导入
                .SetPageLimit(1)                     // 单页扫描
                .SetResultFormats(
                    GmsDocumentScannerOptions.ResultFormatJpeg,
                    GmsDocumentScannerOptions.ResultFormatPdf)
                .SetScannerMode(GmsDocumentScannerOptions.ScannerModeBase) // 基础模式（设备端处理）
                .Build();

            var scanner = GmsDocumentScanning.GetClient(options);

            System.Diagnostics.Debug.WriteLine("[MLKit] 获取扫描 Intent...");

            // 获取扫描 Intent - 使用 GetStartScanIntent
            var intentTask = scanner.GetStartScanIntent(activity);
            var intentSender = await intentTask.AsAsync<AndroidContent.IntentSender>();

            System.Diagnostics.Debug.WriteLine("[MLKit] 启动扫描 Activity...");

            // 创建完成源
            _scanCompletionSource = new TaskCompletionSource<GmsDocumentScanningResult?>();

            // 启动扫描
            var fillInIntent = new AndroidContent.Intent();
            activity.StartIntentSenderForResult(
                intentSender,
                REQUEST_CODE_SCAN,
                fillInIntent,
                0, 0, 0);

            // 等待结果
            var scanningResult = await _scanCompletionSource.Task;

            if (scanningResult == null)
            {
                return ScanResult.Failure("扫描已取消");
            }

            System.Diagnostics.Debug.WriteLine($"[MLKit] 扫描完成，页数: {scanningResult.Pages?.Count ?? 0}");

            // 获取扫描结果
            var pages = scanningResult.Pages;
            if (pages == null || pages.Count == 0)
            {
                return ScanResult.Failure("未扫描到任何页面");
            }

            var firstPage = pages[0];
            var imageUri = firstPage.ImageUri;

            if (imageUri == null)
            {
                return ScanResult.Failure("无法获取扫描图像");
            }

            System.Diagnostics.Debug.WriteLine($"[MLKit] 图像 URI: {imageUri}");

            // 读取图像数据
            using var inputStream = activity.ContentResolver?.OpenInputStream(imageUri);
            if (inputStream == null)
            {
                return ScanResult.Failure("无法读取扫描图像");
            }

            using var memoryStream = new MemoryStream();
            await inputStream.CopyToAsync(memoryStream);
            var imageData = memoryStream.ToArray();

            System.Diagnostics.Debug.WriteLine($"[MLKit] 图像大小: {imageData.Length} 字节");

            // 获取图像尺寸
            var bitmap = AndroidGraphics.BitmapFactory.DecodeByteArray(imageData, 0, imageData.Length);
            int width = bitmap?.Width ?? 0;
            int height = bitmap?.Height ?? 0;
            bitmap?.Recycle();

            return new ScanResult(imageData, width, height, null);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[MLKit] 错误: {ex.GetType().Name} - {ex.Message}");
            System.Diagnostics.Debug.WriteLine($"[MLKit] StackTrace: {ex.StackTrace}");
            return ScanResult.Failure($"ML Kit 扫描失败: {ex.Message}");
        }
    }

    /// <summary>
    /// 处理扫描结果（由 MainActivity 调用）
    /// </summary>
    public static void HandleScanResult(int requestCode, Result resultCode, AndroidContent.Intent? data)
    {
        if (requestCode != REQUEST_CODE_SCAN)
            return;

        System.Diagnostics.Debug.WriteLine($"[MLKit] 收到结果: resultCode={resultCode}");

        if (resultCode == Result.Ok && data != null)
        {
            var result = GmsDocumentScanningResult.FromActivityResultIntent(data);
            _scanCompletionSource?.TrySetResult(result);
        }
        else
        {
            _scanCompletionSource?.TrySetResult(null);
        }
    }
}
