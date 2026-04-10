namespace MauiScan.Controls;

/// <summary>
/// 跨平台相机预览控件
/// </summary>
public class CameraView : View
{
    public event EventHandler<byte[]>? PhotoCaptured;
    public event EventHandler<string>? Error;

    public void OnPhotoCaptured(byte[] data) => PhotoCaptured?.Invoke(this, data);
    public void OnError(string message) => Error?.Invoke(this, message);

    /// <summary>
    /// 拍照命令
    /// </summary>
    public void CapturePhoto()
    {
#if ANDROID
        if (Handler is Platforms.Android.Handlers.CameraPreviewHandler androidHandler)
        {
            androidHandler.CapturePhoto();
        }
#endif
    }
}
