using Foundation;
using MauiScan.Services;
using UIKit;

namespace MauiScan.Platforms.iOS.Services
{
    public class ClipboardService : IClipboardService
    {
        public async Task<bool> CopyImageToClipboardAsync(byte[] imageBytes)
        {
            try
            {
                return await MainThread.InvokeOnMainThreadAsync(() =>
                {
                    try
                    {
                        // 将 byte[] 转换为 UIImage
                        using var data = NSData.FromArray(imageBytes);
                        using var image = UIImage.LoadFromData(data);

                        if (image == null)
                        {
                            return false;
                        }

                        // 复制到剪贴板
                        UIPasteboard.General.Image = image;
                        return true;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error copying image to clipboard: {ex}");
                        return false;
                    }
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CopyImageToClipboardAsync exception: {ex}");
                return false;
            }
        }

        public async Task<bool> CopyFilePathToClipboardAsync(string filePath)
        {
            try
            {
                return await MainThread.InvokeOnMainThreadAsync(() =>
                {
                    try
                    {
                        if (string.IsNullOrEmpty(filePath))
                        {
                            return false;
                        }

                        // iOS 上复制文件路径（作为 URL）
                        var url = NSUrl.FromFilename(filePath);
                        if (url != null)
                        {
                            UIPasteboard.General.Url = url;
                            return true;
                        }

                        // 备用方案：复制为纯文本
                        UIPasteboard.General.String = filePath;
                        return true;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error copying file path to clipboard: {ex}");
                        return false;
                    }
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CopyFilePathToClipboardAsync exception: {ex}");
                return false;
            }
        }
    }
}
