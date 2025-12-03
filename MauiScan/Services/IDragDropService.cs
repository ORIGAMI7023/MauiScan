namespace MauiScan.Services;

/// <summary>
/// 拖放服务接口（跨平台）
/// </summary>
public interface IDragDropService
{
    /// <summary>
    /// 开始拖动图片
    /// </summary>
    /// <param name="view">要拖动的视图</param>
    /// <param name="imageBytes">图片数据</param>
    /// <returns>是否成功开始拖动</returns>
    Task<bool> StartDragImageAsync(IView view, byte[] imageBytes);
}
