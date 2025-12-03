using Android.Content;
using Android.Views;
using MauiScan.Services;
using AndroidUri = Android.Net.Uri;
using AndroidApp = Android.App.Application;
using AndroidView = Android.Views.View;

namespace MauiScan.Platforms.Android.Services;

/// <summary>
/// Android 拖放服务实现
/// </summary>
public class DragDropService : IDragDropService
{
    public async Task<bool> StartDragImageAsync(IView view, byte[] imageBytes)
    {
        try
        {
            // 保存图片到临时文件
            var tempPath = System.IO.Path.Combine(FileSystem.CacheDirectory, $"drag_{DateTime.Now:yyyyMMddHHmmss}.jpg");
            await File.WriteAllBytesAsync(tempPath, imageBytes);

            // 获取 Content URI
            var file = new Java.IO.File(tempPath);
            var context = AndroidApp.Context;
            var authority = $"{context.PackageName}.fileprovider";

            AndroidUri? contentUri = null;
            try
            {
                contentUri = AndroidX.Core.Content.FileProvider.GetUriForFile(context, authority, file);
            }
            catch
            {
                contentUri = AndroidUri.FromFile(file);
            }

            // 获取原生 Android View
            var handler = view.Handler;
            if (handler?.PlatformView is not AndroidView androidView)
            {
                System.Diagnostics.Debug.WriteLine("[DragDrop] 无法获取 Android View");
                return false;
            }

            // 在主线程上启动拖放
            MainThread.BeginInvokeOnMainThread(() =>
            {
                try
                {
                    // 创建 ClipData
                    var clipData = ClipData.NewUri(context.ContentResolver, "扫描图像", contentUri);

                    // 创建拖动阴影
                    var shadowBuilder = new AndroidView.DragShadowBuilder(androidView);

                    // 设置拖动标志：允许跨应用 + 授予 URI 读取权限
                    var flags = (int)DragFlags.Global | (int)DragFlags.GlobalUriRead;

                    // 开始拖放
                    androidView.StartDragAndDrop(clipData, shadowBuilder, null, flags);

                    System.Diagnostics.Debug.WriteLine($"[DragDrop] 拖放已启动: {contentUri}");
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"[DragDrop] 启动拖放失败: {ex.Message}");
                }
            });

            return true;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[DragDrop] 准备拖放失败: {ex.Message}");
            return false;
        }
    }
}
