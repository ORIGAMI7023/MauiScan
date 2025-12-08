using Microsoft.Extensions.Logging;
using MauiScan.Services;
using MauiScan.Services.Sync;
using MauiScan.Views;
using MauiScan.Controls;
using MauiScan.ML.Services;
using MauiScan.ML.Models;

namespace MauiScan
{
    public static class MauiProgram
    {
        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                })
                .ConfigureMauiHandlers(handlers =>
                {
#if ANDROID
                    handlers.AddHandler<CameraView, Platforms.Android.Handlers.CameraPreviewHandler>();
#endif
                });

            // 注册服务（使用 Native C++ OpenCV 实现）
            builder.Services.AddSingleton<IImageProcessingService, NativeImageProcessingService>();

            // 注册图像裁剪服务
            builder.Services.AddSingleton<ImageCropService>();

            // 注册两阶段检测服务
            builder.Services.AddSingleton<TwoStageDetectionService>();

            // 注册 ML 推理服务
            builder.Services.AddSingleton<IMLInferenceService>(sp =>
            {
                var modelPath = Path.Combine(FileSystem.AppDataDirectory, "ppt_corner_detector.onnx");
                var config = new ModelConfig
                {
                    InputWidth = 512,
                    InputHeight = 512,
                    HighQualityThreshold = 0.85f,
                    MediumQualityThreshold = 0.60f,
                    EnableGpuAcceleration = false  // 暂时禁用 GPU
                };
                return new OnnxInferenceService(modelPath, config);
            });

            // 注册同步服务
            builder.Services.AddSingleton<ScanSyncService>(sp =>
                new ScanSyncService("https://mauiscan.origami7023.net.cn"));

            // 注册平台特定服务
#if ANDROID
            builder.Services.AddSingleton<ICameraService, Platforms.Android.Services.CameraService>();
            builder.Services.AddSingleton<IClipboardService, Platforms.Android.Services.ClipboardService>();
            builder.Services.AddSingleton<IDragDropService, Platforms.Android.Services.DragDropService>();
            builder.Services.AddSingleton<IManualAnnotationService, Platforms.Android.Services.ManualAnnotationService>();
#endif

            // 注册页面
            builder.Services.AddTransient<ScanPage>();
            builder.Services.AddTransient<CameraPage>();
            builder.Services.AddTransient<HistoryPage>();
            builder.Services.AddTransient<MLTestPage>();  // ML 测试页面
            builder.Services.AddTransient<TwoStageDetectionTestPage>();  // 两阶段检测测试页面

#if DEBUG
    		builder.Logging.AddDebug();
#endif

            return builder.Build();
        }
    }
}
