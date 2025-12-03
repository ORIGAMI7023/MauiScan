namespace MauiScan.Services;

/// <summary>
/// 剪贴板服务接口（跨平台）
/// </summary>
public interface IClipboardService
{
    /// <summary>
    /// 将图像复制到系统剪贴板
    /// </summary>
    /// <param name="imageBytes">图像字节数据</param>
    /// <returns>是否成功</returns>
    Task<bool> CopyImageToClipboardAsync(byte[] imageBytes);

    /// <summary>
    /// 将文件路径复制到剪贴板（用于分享）
    /// </summary>
    Task<bool> CopyFilePathToClipboardAsync(string filePath);
}
