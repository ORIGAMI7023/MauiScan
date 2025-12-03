using Android.Content;
using Android.Graphics;
using Android.Hardware.Camera2;
using Android.Hardware.Camera2.Params;
using Android.Media;
using Android.OS;
using Android.Views;
using Microsoft.Maui.Handlers;
using MauiScan.Controls;
using Size = Android.Util.Size;

namespace MauiScan.Platforms.Android.Handlers;

/// <summary>
/// Camera2 相机预览 Handler
/// </summary>
public class CameraPreviewHandler : ViewHandler<CameraView, TextureView>
{
    private CameraDevice? _cameraDevice;
    private CameraCaptureSession? _captureSession;
    private CaptureRequest.Builder? _previewRequestBuilder;
    private ImageReader? _imageReader;
    private Handler? _backgroundHandler;
    private HandlerThread? _backgroundThread;
    private Size? _previewSize;
    private string? _cameraId;
    private bool _isCapturing;

    public static IPropertyMapper<CameraView, CameraPreviewHandler> Mapper =
        new PropertyMapper<CameraView, CameraPreviewHandler>(ViewHandler.ViewMapper);

    public CameraPreviewHandler() : base(Mapper) { }

    protected override TextureView CreatePlatformView()
    {
        var textureView = new TextureView(Context!);
        textureView.SurfaceTextureListener = new SurfaceTextureListener(this);
        return textureView;
    }

    protected override void ConnectHandler(TextureView platformView)
    {
        base.ConnectHandler(platformView);
        StartBackgroundThread();
    }

    protected override void DisconnectHandler(TextureView platformView)
    {
        CloseCamera();
        StopBackgroundThread();
        base.DisconnectHandler(platformView);
    }

    private void StartBackgroundThread()
    {
        _backgroundThread = new HandlerThread("CameraBackground");
        _backgroundThread.Start();
        _backgroundHandler = new Handler(_backgroundThread.Looper!);
    }

    private void StopBackgroundThread()
    {
        _backgroundThread?.QuitSafely();
        try
        {
            _backgroundThread?.Join();
            _backgroundThread = null;
            _backgroundHandler = null;
        }
        catch { }
    }

    private void OpenCamera(int width, int height)
    {
        try
        {
            var manager = (CameraManager)Context!.GetSystemService(Context.CameraService)!;

            // 找到后置摄像头
            foreach (var cameraId in manager.GetCameraIdList()!)
            {
                var characteristics = manager.GetCameraCharacteristics(cameraId);
                var facing = (int)(characteristics.Get(CameraCharacteristics.LensFacing) ?? 0);

                if (facing == (int)LensFacing.Back)
                {
                    _cameraId = cameraId;

                    // 获取支持的预览尺寸
                    var map = (StreamConfigurationMap)characteristics.Get(CameraCharacteristics.ScalerStreamConfigurationMap)!;
                    var outputSizes = map.GetOutputSizes(Java.Lang.Class.FromType(typeof(SurfaceTexture)));
                    _previewSize = ChooseOptimalSize(outputSizes!, width, height);

                    System.Diagnostics.Debug.WriteLine($"[Camera2] 预览尺寸: {_previewSize.Width}x{_previewSize.Height}");
                    break;
                }
            }

            if (_cameraId == null)
            {
                VirtualView?.OnError("未找到后置摄像头");
                return;
            }

            manager.OpenCamera(_cameraId, new CameraStateCallback(this), _backgroundHandler);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[Camera2] 打开相机失败: {ex.Message}");
            VirtualView?.OnError($"打开相机失败: {ex.Message}");
        }
    }

    private Size ChooseOptimalSize(Size[] choices, int width, int height)
    {
        var targetRatio = (double)width / height;
        Size? result = null;
        double minDiff = double.MaxValue;

        foreach (var size in choices)
        {
            var ratio = (double)size.Width / size.Height;
            var diff = Math.Abs(ratio - targetRatio);

            if (diff < minDiff && size.Width <= 1920)
            {
                minDiff = diff;
                result = size;
            }
        }

        return result ?? choices[0];
    }

    private void CreateCameraPreviewSession()
    {
        try
        {
            var texture = PlatformView!.SurfaceTexture!;
            texture.SetDefaultBufferSize(_previewSize!.Width, _previewSize.Height);

            var surface = new Surface(texture);

            // 创建 ImageReader 用于拍照
            _imageReader = ImageReader.NewInstance(_previewSize.Width, _previewSize.Height, ImageFormatType.Jpeg, 1);
            _imageReader.SetOnImageAvailableListener(new ImageAvailableListener(this), _backgroundHandler);

            _previewRequestBuilder = _cameraDevice!.CreateCaptureRequest(CameraTemplate.Preview);
            _previewRequestBuilder.AddTarget(surface);

            var surfaces = new List<Surface> { surface, _imageReader.Surface! };

            _cameraDevice.CreateCaptureSession(surfaces, new CaptureSessionCallback(this, surface), _backgroundHandler);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[Camera2] 创建预览会话失败: {ex.Message}");
            VirtualView?.OnError($"创建预览失败: {ex.Message}");
        }
    }

    public void CapturePhoto()
    {
        if (_isCapturing || _cameraDevice == null || _captureSession == null)
            return;

        _isCapturing = true;

        try
        {
            var captureBuilder = _cameraDevice.CreateCaptureRequest(CameraTemplate.StillCapture);
            captureBuilder!.AddTarget(_imageReader!.Surface!);

            // 自动对焦和自动曝光
            captureBuilder.Set(CaptureRequest.ControlAfMode!, (int)ControlAFMode.ContinuousPicture);
            captureBuilder.Set(CaptureRequest.ControlAeMode!, (int)ControlAEMode.OnAutoFlash);

            // 设置 JPEG 方向
            captureBuilder.Set(CaptureRequest.JpegOrientation!, 90);

            _captureSession.Capture(captureBuilder.Build(), new CaptureCallback(this), _backgroundHandler);
        }
        catch (Exception ex)
        {
            _isCapturing = false;
            System.Diagnostics.Debug.WriteLine($"[Camera2] 拍照失败: {ex.Message}");
            VirtualView?.OnError($"拍照失败: {ex.Message}");
        }
    }

    private void CloseCamera()
    {
        _captureSession?.Close();
        _captureSession = null;
        _cameraDevice?.Close();
        _cameraDevice = null;
        _imageReader?.Close();
        _imageReader = null;
    }

    #region Callbacks

    private class SurfaceTextureListener : Java.Lang.Object, TextureView.ISurfaceTextureListener
    {
        private readonly CameraPreviewHandler _handler;

        public SurfaceTextureListener(CameraPreviewHandler handler) => _handler = handler;

        public void OnSurfaceTextureAvailable(SurfaceTexture surface, int width, int height)
        {
            _handler.OpenCamera(width, height);
        }

        public bool OnSurfaceTextureDestroyed(SurfaceTexture surface)
        {
            _handler.CloseCamera();
            return true;
        }

        public void OnSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) { }
        public void OnSurfaceTextureUpdated(SurfaceTexture surface) { }
    }

    private class CameraStateCallback : CameraDevice.StateCallback
    {
        private readonly CameraPreviewHandler _handler;

        public CameraStateCallback(CameraPreviewHandler handler) => _handler = handler;

        public override void OnOpened(CameraDevice camera)
        {
            System.Diagnostics.Debug.WriteLine("[Camera2] 相机已打开");
            _handler._cameraDevice = camera;
            _handler.CreateCameraPreviewSession();
        }

        public override void OnDisconnected(CameraDevice camera)
        {
            camera.Close();
            _handler._cameraDevice = null;
        }

        public override void OnError(CameraDevice camera, CameraError error)
        {
            camera.Close();
            _handler._cameraDevice = null;
            _handler.VirtualView?.OnError($"相机错误: {error}");
        }
    }

    private class CaptureSessionCallback : CameraCaptureSession.StateCallback
    {
        private readonly CameraPreviewHandler _handler;
        private readonly Surface _surface;

        public CaptureSessionCallback(CameraPreviewHandler handler, Surface surface)
        {
            _handler = handler;
            _surface = surface;
        }

        public override void OnConfigured(CameraCaptureSession session)
        {
            System.Diagnostics.Debug.WriteLine("[Camera2] 预览会话已配置");
            _handler._captureSession = session;

            try
            {
                _handler._previewRequestBuilder!.Set(CaptureRequest.ControlAfMode!, (int)ControlAFMode.ContinuousPicture);
                _handler._previewRequestBuilder.Set(CaptureRequest.ControlAeMode!, (int)ControlAEMode.OnAutoFlash);

                session.SetRepeatingRequest(_handler._previewRequestBuilder.Build(), null, _handler._backgroundHandler);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[Camera2] 启动预览失败: {ex.Message}");
            }
        }

        public override void OnConfigureFailed(CameraCaptureSession session)
        {
            _handler.VirtualView?.OnError("配置相机会话失败");
        }
    }

    private class CaptureCallback : CameraCaptureSession.CaptureCallback
    {
        private readonly CameraPreviewHandler _handler;

        public CaptureCallback(CameraPreviewHandler handler) => _handler = handler;

        public override void OnCaptureCompleted(CameraCaptureSession session, CaptureRequest request, TotalCaptureResult result)
        {
            System.Diagnostics.Debug.WriteLine("[Camera2] 拍照完成");
        }
    }

    private class ImageAvailableListener : Java.Lang.Object, ImageReader.IOnImageAvailableListener
    {
        private readonly CameraPreviewHandler _handler;

        public ImageAvailableListener(CameraPreviewHandler handler) => _handler = handler;

        public void OnImageAvailable(ImageReader? reader)
        {
            if (reader == null) return;

            try
            {
                using var image = reader.AcquireLatestImage();
                if (image == null) return;

                var buffer = image.GetPlanes()![0].Buffer!;
                var bytes = new byte[buffer.Remaining()];
                buffer.Get(bytes);

                System.Diagnostics.Debug.WriteLine($"[Camera2] 图像捕获成功: {bytes.Length} 字节");

                _handler._isCapturing = false;
                _handler.VirtualView?.OnPhotoCaptured(bytes);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[Camera2] 读取图像失败: {ex.Message}");
                _handler._isCapturing = false;
            }
        }
    }

    #endregion
}
