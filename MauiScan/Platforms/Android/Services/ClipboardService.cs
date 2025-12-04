using Android.Content;
using Android.Graphics;
using MauiScan.Services;
using AndroidUri = Android.Net.Uri;
using AndroidApp = Android.App.Application;

namespace MauiScan.Platforms.Android.Services;

/// <summary>
/// Android 剪贴板服务实现
/// </summary>
public class ClipboardService : IClipboardService
{
    public async Task<bool> CopyImageToClipboardAsync(byte[] imageBytes)
    {
        try
        {
            // 将图像保存到临时文件
            var tempPath = System.IO.Path.Combine(FileSystem.CacheDirectory, $"scan_{DateTime.Now:yyyyMMddHHmmss}.jpg");
            await File.WriteAllBytesAsync(tempPath, imageBytes);

            // 获取 Content URI（Android 7.0+ 需要使用 FileProvider）
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
                // 如果 FileProvider 未配置，降级为直接路径（仅用于测试）
                contentUri = AndroidUri.FromFile(file);
            }

            // 复制到剪贴板（使用明确的 MIME 类型）
            var clipboardManager = (ClipboardManager?)context.GetSystemService(Context.ClipboardService);
            if (clipboardManager == null)
                return false;

            // 创建带有明确 MIME 类型的 ClipData
            var mimeTypes = new string[] { "image/jpeg" };
            var clipDescription = new ClipDescription("扫描图像", mimeTypes);
            var item = new ClipData.Item(contentUri);
            var clip = new ClipData(clipDescription, item);

            // 授予所有应用临时读取权限
            context.GrantUriPermission("*", contentUri, ActivityFlags.GrantReadUriPermission);

            clipboardManager.PrimaryClip = clip;

            return true;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"复制到剪贴板失败: {ex.Message}");
            return false;
        }
    }

    public async Task<bool> CopyFilePathToClipboardAsync(string filePath)
    {
        try
        {
            await Clipboard.Default.SetTextAsync(filePath);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
