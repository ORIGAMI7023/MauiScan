using Microsoft.Extensions.Logging;
using MauiScan.Services;
using MauiScan.Services.Sync;
using MauiScan.Views;
using MauiScan.Controls;

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

            // 注册同步服务
            builder.Services.AddSingleton<ScanSyncService>(sp =>
                new ScanSyncService("https://mauiscan.origami7023.net.cn"));

            // 注册平台特定服务
#if ANDROID
            builder.Services.AddSingleton<ICameraService, Platforms.Android.Services.CameraService>();
            builder.Services.AddSingleton<IClipboardService, Platforms.Android.Services.ClipboardService>();
            builder.Services.AddSingleton<IDragDropService, Platforms.Android.Services.DragDropService>();
            builder.Services.AddSingleton<IManualAnnotationService, Platforms.Android.Services.ManualAnnotationService>();
#elif __IOS__
            builder.Services.AddSingleton<ICameraService, Platforms.iOS.Services.CameraService>();
            builder.Services.AddSingleton<IClipboardService, Platforms.iOS.Services.ClipboardService>();
#else
            // 其他平台暂不实现
#endif

            // 注册页面
            builder.Services.AddTransient<ScanPage>();
            builder.Services.AddTransient<CameraPage>();
            builder.Services.AddTransient<HistoryPage>();

#if DEBUG
    		builder.Logging.AddDebug();
#endif

            return builder.Build();
        }
    }
}
