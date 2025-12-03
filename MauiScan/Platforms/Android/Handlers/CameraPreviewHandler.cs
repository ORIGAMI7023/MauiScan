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
using Rect = Android.Graphics.Rect;

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
    private Size? _captureSize;
    private string? _cameraId;
    private bool _isCapturing;
    private int _sensorOrientation;

    // 缩放相关
    private ScaleGestureDetector? _scaleGestureDetector;
    private float _maxZoom = 1f;
    private float _currentZoom = 1f;
    private Rect? _sensorArraySize;
    private ScaleListener? _scaleListener;

    public static IPropertyMapper<CameraView, CameraPreviewHandler> Mapper =
        new PropertyMapper<CameraView, CameraPreviewHandler>(ViewHandler.ViewMapper);

    public CameraPreviewHandler() : base(Mapper) { }

    protected override TextureView CreatePlatformView()
    {
        var textureView = new TextureView(Context!);
        textureView.SurfaceTextureListener = new SurfaceTextureListener(this);

        // 初始化缩放手势检测器
        _scaleListener = new ScaleListener(this);
        _scaleGestureDetector = new ScaleGestureDetector(Context!, _scaleListener);

        // 设置触摸监听（仅处理缩放）
        textureView.Touch += (sender, e) =>
        {
            _scaleGestureDetector?.OnTouchEvent(e.Event!);
            e.Handled = true;
        };

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

                    // 获取传感器方向
                    _sensorOrientation = (int)(characteristics.Get(CameraCharacteristics.SensorOrientation) ?? 90);

                    // 获取最大缩放倍数
                    var maxZoomObj = characteristics.Get(CameraCharacteristics.ScalerAvailableMaxDigitalZoom);
                    _maxZoom = maxZoomObj != null ? (float)maxZoomObj : 1f;
                    _currentZoom = 1f;

                    // 获取传感器尺寸（用于计算缩放裁剪区域）
                    _sensorArraySize = characteristics.Get(CameraCharacteristics.SensorInfoActiveArraySize) as Rect;

                    // 获取支持的尺寸
                    var map = (StreamConfigurationMap)characteristics.Get(CameraCharacteristics.ScalerStreamConfigurationMap)!;
                    var outputSizes = map.GetOutputSizes(Java.Lang.Class.FromType(typeof(global::Android.Graphics.SurfaceTexture)));
                    var jpegSizes = map.GetOutputSizes((int)ImageFormatType.Jpeg);

                    // 选择预览尺寸（适配屏幕）
                    _previewSize = ChooseOptimalPreviewSize(outputSizes!, width, height);
                    // 选择拍照尺寸（最高分辨率）
                    _captureSize = ChooseLargestSize(jpegSizes!);

                    System.Diagnostics.Debug.WriteLine($"[Camera2] 预览尺寸: {_previewSize.Width}x{_previewSize.Height}");
                    System.Diagnostics.Debug.WriteLine($"[Camera2] 拍照尺寸: {_captureSize.Width}x{_captureSize.Height}");
                    System.Diagnostics.Debug.WriteLine($"[Camera2] 传感器方向: {_sensorOrientation}");
                    System.Diagnostics.Debug.WriteLine($"[Camera2] 最大缩放: {_maxZoom}x");
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

    private Size ChooseOptimalPreviewSize(Size[] choices, int viewWidth, int viewHeight)
    {
        // 屏幕是竖屏，但相机传感器是横向的，所以需要交换宽高比较
        var targetRatio = (double)Math.Max(viewWidth, viewHeight) / Math.Min(viewWidth, viewHeight);

        Size? result = null;
        double minDiff = double.MaxValue;

        foreach (var size in choices)
        {
            var ratio = (double)size.Width / size.Height;
            var diff = Math.Abs(ratio - targetRatio);

            // 选择宽高比接近且尺寸适中的预览尺寸
            if (diff < minDiff && size.Width <= 1920 && size.Width >= 640)
            {
                minDiff = diff;
                result = size;
            }
        }

        // 如果没找到合适的，选择第一个不超过 1920 的
        if (result == null)
        {
            foreach (var size in choices)
            {
                if (size.Width <= 1920)
                {
                    result = size;
                    break;
                }
            }
        }

        return result ?? choices[0];
    }

    private Size ChooseLargestSize(Size[] choices)
    {
        Size? largest = null;
        long maxPixels = 0;

        foreach (var size in choices)
        {
            long pixels = (long)size.Width * size.Height;
            // 限制最大 4000 万像素，避免内存问题
            if (pixels > maxPixels && pixels <= 40_000_000)
            {
                maxPixels = pixels;
                largest = size;
            }
        }

        return largest ?? choices[0];
    }

    private void CreateCameraPreviewSession()
    {
        try
        {
            var texture = PlatformView!.SurfaceTexture!;
            texture.SetDefaultBufferSize(_previewSize!.Width, _previewSize.Height);

            var surface = new global::Android.Views.Surface(texture);

            // 创建 ImageReader 用于拍照（使用高分辨率）
            _imageReader = ImageReader.NewInstance(_captureSize!.Width, _captureSize.Height, ImageFormatType.Jpeg, 1);
            _imageReader.SetOnImageAvailableListener(new ImageAvailableListener(this), _backgroundHandler);

            _previewRequestBuilder = _cameraDevice!.CreateCaptureRequest(CameraTemplate.Preview);
            _previewRequestBuilder.AddTarget(surface);

            var surfaces = new List<global::Android.Views.Surface> { surface, _imageReader.Surface! };

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

            // 设置 JPEG 方向（根据传感器方向）
            captureBuilder.Set(CaptureRequest.JpegOrientation!, _sensorOrientation);

            // 应用当前缩放
            var cropRect = GetZoomRect(_currentZoom);
            if (cropRect != null)
            {
                captureBuilder.Set(CaptureRequest.ScalerCropRegion!, cropRect);
            }

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

    internal void ApplyZoom(float scaleFactor)
    {
        if (_sensorArraySize == null || _previewRequestBuilder == null || _captureSession == null)
            return;

        // 计算新的缩放值
        _currentZoom = Math.Clamp(_currentZoom * scaleFactor, 1f, _maxZoom);

        // 计算裁剪区域
        var cropRect = GetZoomRect(_currentZoom);
        if (cropRect != null)
        {
            _previewRequestBuilder.Set(CaptureRequest.ScalerCropRegion!, cropRect);

            try
            {
                _captureSession.SetRepeatingRequest(_previewRequestBuilder.Build(), null, _backgroundHandler);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[Camera2] 缩放失败: {ex.Message}");
            }
        }
    }

    private Rect? GetZoomRect(float zoomLevel)
    {
        if (_sensorArraySize == null) return null;

        var sensorWidth = _sensorArraySize.Width();
        var sensorHeight = _sensorArraySize.Height();

        var cropWidth = (int)(sensorWidth / zoomLevel);
        var cropHeight = (int)(sensorHeight / zoomLevel);

        var left = (sensorWidth - cropWidth) / 2;
        var top = (sensorHeight - cropHeight) / 2;

        return new Rect(left, top, left + cropWidth, top + cropHeight);
    }

    #region Callbacks

    private class SurfaceTextureListener : Java.Lang.Object, TextureView.ISurfaceTextureListener
    {
        private readonly CameraPreviewHandler _handler;

        public SurfaceTextureListener(CameraPreviewHandler handler) => _handler = handler;

        public void OnSurfaceTextureAvailable(global::Android.Graphics.SurfaceTexture surface, int width, int height)
        {
            _handler.OpenCamera(width, height);
        }

        public bool OnSurfaceTextureDestroyed(global::Android.Graphics.SurfaceTexture surface)
        {
            _handler.CloseCamera();
            return true;
        }

        public void OnSurfaceTextureSizeChanged(global::Android.Graphics.SurfaceTexture surface, int width, int height) { }
        public void OnSurfaceTextureUpdated(global::Android.Graphics.SurfaceTexture surface) { }
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
        private readonly global::Android.Views.Surface _surface;

        public CaptureSessionCallback(CameraPreviewHandler handler, global::Android.Views.Surface surface)
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

    private class ScaleListener : Java.Lang.Object, ScaleGestureDetector.IOnScaleGestureListener
    {
        private readonly CameraPreviewHandler _handler;

        public ScaleListener(CameraPreviewHandler handler) => _handler = handler;

        public bool OnScale(ScaleGestureDetector detector)
        {
            _handler.ApplyZoom(detector.ScaleFactor);
            return true;
        }

        public bool OnScaleBegin(ScaleGestureDetector detector) => true;

        public void OnScaleEnd(ScaleGestureDetector detector) { }
    }

    #endregion
}
