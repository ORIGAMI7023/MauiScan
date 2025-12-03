using Microsoft.Extensions.Logging;
using MauiScan.Services;
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

#if ANDROID
            builder.Services.AddSingleton<ICameraService, Platforms.Android.Services.CameraService>();
            builder.Services.AddSingleton<IClipboardService, Platforms.Android.Services.ClipboardService>();
            builder.Services.AddSingleton<IDragDropService, Platforms.Android.Services.DragDropService>();
#elif IOS
            // iOS: 暂未实现，需要在 Mac 上开发
            // builder.Services.AddSingleton<ICameraService, Platforms.iOS.Services.CameraService>();
            // builder.Services.AddSingleton<IClipboardService, Platforms.iOS.Services.ClipboardService>();
#else
            // 其他平台暂不实现
#endif

            // 注册页面
            builder.Services.AddTransient<ScanPage>();
            builder.Services.AddTransient<CameraPage>();

#if DEBUG
    		builder.Logging.AddDebug();
#endif

            return builder.Build();
        }
    }
}
