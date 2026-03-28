using Android.Animation;
using Android.Content;
using Android.Graphics;
using Android.OS;
using Android.Runtime;
using Android.Util;
using Android.Views;
using Android.Widget;
using AndroidX.Camera.Core;
using AndroidX.Camera.Lifecycle;
using AndroidX.Lifecycle;
using Java.Util.Concurrent;
using Microsoft.Maui.Handlers;
using MauiScan.Controls;
using Rect = Android.Graphics.Rect;

namespace MauiScan.Platforms.Android.Handlers;

/// <summary>
/// CameraX 相机预览 Handler
/// </summary>
public class CameraPreviewHandler : ViewHandler<CameraView, FrameLayout>
{
    private TextureView? _textureView;
    private FocusOverlayView? _focusOverlay;

    // CameraX
    private ProcessCameraProvider? _cameraProvider;
    private Preview? _preview;
    private ImageCapture? _imageCapture;
    private ICamera? _camera;

    // 缩放相关
    private ScaleGestureDetector? _scaleGestureDetector;
    private ScaleListener? _scaleListener;

    // 设备物理方向监听
    private DeviceOrientationListener? _orientationListener;

    public static IPropertyMapper<CameraView, CameraPreviewHandler> Mapper =
        new PropertyMapper<CameraView, CameraPreviewHandler>(ViewHandler.ViewMapper);

    public CameraPreviewHandler() : base(Mapper) { }

    protected override FrameLayout CreatePlatformView()
    {
        var frameLayout = new FrameLayout(Context!)
        {
            LayoutParameters = new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.MatchParent)
        };

        _textureView = new TextureView(Context!);
        _textureView.SurfaceTextureListener = new SurfaceTextureListener(this);
        _textureView.LayoutParameters = new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.MatchParent);
        frameLayout.AddView(_textureView);

        _focusOverlay = new FocusOverlayView(Context!);
        _focusOverlay.LayoutParameters = new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.MatchParent);
        frameLayout.AddView(_focusOverlay);

        // 初始化缩放手势检测器
        _scaleListener = new ScaleListener(this);
        _scaleGestureDetector = new ScaleGestureDetector(Context!, _scaleListener);

        // 设置触摸监听（处理点击对焦/测光和缩放）
        var tapGestureDetector = new GestureDetector(Context!, new TapListener(this));
        _textureView.Touch += (sender, e) =>
        {
            _scaleGestureDetector?.OnTouchEvent(e.Event!);
            tapGestureDetector.OnTouchEvent(e.Event!);
            e.Handled = true;
        };

        return frameLayout;
    }

    protected override async void ConnectHandler(FrameLayout platformView)
    {
        base.ConnectHandler(platformView);
        StartOrientationListener();
        await StartCameraAsync();
    }

    protected override void DisconnectHandler(FrameLayout platformView)
    {
        StopOrientationListener();
        _cameraProvider?.UnbindAll();
        _cameraProvider = null;
        _camera = null;
        _preview = null;
        _imageCapture = null;
        base.DisconnectHandler(platformView);
    }

    private void StartOrientationListener()
    {
        _orientationListener = new DeviceOrientationListener(Context!, this);
        if (_orientationListener.CanDetectOrientation())
        {
            _orientationListener.Enable();
        }
    }

    private void StopOrientationListener()
    {
        _orientationListener?.Disable();
        _orientationListener = null;
    }

    internal void OnOrientationChanged(int orientation)
    {
        if (orientation == OrientationEventListener.OrientationUnknown)
            return;
    }

    private async Task StartCameraAsync()
    {
        try
        {
            // 获取 ProcessCameraProvider
            // getInstance 在 Kotlin 中定义为 ProcessCameraProvider 的扩展函数
            // 编译后在 Companion 类上作为成员方法
            // 通过 Companion 单例实例调用
            var providerClass = Java.Lang.Class.ForName("androidx.camera.lifecycle.ProcessCameraProvider");
            var contextClass = Java.Lang.Class.ForName("android.content.Context");

            // 获取 Companion 单例
            var companionField = providerClass!.GetDeclaredField("Companion");
            companionField.Accessible = true;
            var companion = companionField.Get(null)!;

            // 在 Companion 类上调用 getInstance(context)
            // Kotlin 扩展函数在 JVM 中作为 Companion 的成员方法
            var companionClass = companion!.Class!;
            var method = companionClass.GetMethod("getInstance", contextClass);
            var future = method.Invoke(companion, new[] { Context! });
            System.Diagnostics.Debug.WriteLine("[CameraX] getInstance 调用成功");

            // ListenableFuture 继承自 Future，直接 Get() 阻塞等待
            if (future is Java.Util.Concurrent.IFuture javaFuture)
            {
                _cameraProvider = javaFuture.Get() as ProcessCameraProvider;
            }

            if (_cameraProvider == null)
            {
                VirtualView?.OnError("获取 CameraProvider 失败");
                return;
            }

            // 预览
            _preview = new Preview.Builder().Build();
            var executor = Context!.MainExecutor!;
            _preview.SetSurfaceProvider(executor, new PreviewSurfaceProvider(this));

            // 拍照（最高质量）
            _imageCapture = new ImageCapture.Builder()
                .SetCaptureMode(ImageCapture.CaptureModeMaximizeQuality)
                .Build();

            // 获取 LifecycleOwner
            var lifecycleOwner = GetLifecycleOwner();
            if (lifecycleOwner == null)
            {
                VirtualView?.OnError("获取 LifecycleOwner 失败");
                return;
            }

            // 绑定到生命周期
            _camera = _cameraProvider.BindToLifecycle(
                lifecycleOwner,
                CameraSelector.DefaultBackCamera,
                _preview,
                _imageCapture
            );

            System.Diagnostics.Debug.WriteLine("[CameraX] 相机已启动");
        }
        catch (System.Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[CameraX] 启动相机失败: {ex.Message}");
            VirtualView?.OnError($"启动相机失败: {ex.Message}");
        }
    }

    private ILifecycleOwner? GetLifecycleOwner()
    {
        if (Context is AndroidX.Activity.ComponentActivity componentActivity)
            return componentActivity;

        var activity = Platform.CurrentActivity;
        if (activity is AndroidX.Activity.ComponentActivity ca)
            return ca;

        return null;
    }

    public void CapturePhoto()
    {
        if (_imageCapture == null)
            return;

        try
        {
            _imageCapture.TakePicture(
                Context!.MainExecutor!,
                new ImageCaptureCallback(this)
            );
        }
        catch (System.Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[CameraX] 拍照失败: {ex.Message}");
            VirtualView?.OnError($"拍照失败: {ex.Message}");
        }
    }

    internal void FocusAndMeter(float x, float y)
    {
        if (_camera == null || _textureView == null)
            return;

        int actualWidth = _textureView.Width;
        int actualHeight = _textureView.Height;

        if (actualWidth == 0 || actualHeight == 0)
            return;

        float normalizedX = x / actualWidth;
        float normalizedY = y / actualHeight;

        // SurfaceOrientedMeteringPointFactory 使用 Surface 坐标系
        var factory = new SurfaceOrientedMeteringPointFactory(actualWidth, actualHeight);
        var point = factory.CreatePoint(normalizedX, normalizedY);

        var action = new FocusMeteringAction.Builder(point)
            .Build();

        System.Diagnostics.Debug.WriteLine($"[CameraX] 点击对焦: ({x:F0},{y:F0}) 归一化=({normalizedX:F3},{normalizedY:F3})");

        try
        {
            var focusFuture = _camera.CameraControl.StartFocusAndMetering(action);
            if (focusFuture != null)
            {
                focusFuture.AddListener(
                    new Java.Lang.Runnable(() =>
                    {
                        try
                        {
                            var result = focusFuture.Get();
                            System.Diagnostics.Debug.WriteLine($"[CameraX] 对焦结果: {result}");
                        }
                        catch (System.Exception ex)
                        {
                            System.Diagnostics.Debug.WriteLine($"[CameraX] 对焦结果异常: {ex.Message}");
                        }
                    }),
                    Context!.MainExecutor!
                );
            }
        }
        catch (System.Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[CameraX] 对焦失败: {ex.Message}");
        }
    }

    internal void ApplyZoom(float scaleFactor)
    {
        if (_camera == null)
            return;

        try
        {
            var zoomState = _camera.CameraInfo.ZoomState;
            var value = zoomState?.Value;
            if (value == null) return;

            // ZoomState 属性通过 JNI Invoker 获取
            float currentRatio = GetZoomProperty(value, "ZoomRatio");
            float minRatio = GetZoomProperty(value, "MinZoomRatio");
            float maxRatio = GetZoomProperty(value, "MaxZoomRatio");

            var newRatio = System.Math.Clamp(currentRatio * scaleFactor, minRatio, maxRatio);
            _camera.CameraControl.SetZoomRatio(newRatio);
        }
        catch (System.Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[CameraX] 缩放失败: {ex.Message}");
        }
    }

    private static float GetZoomProperty(Java.Lang.Object zoomState, string propertyName)
    {
        try
        {
            var getMethod = zoomState.GetType().GetMethod($"get{propertyName}")!;
            var result = getMethod.Invoke(zoomState, null);
            return result is Java.Lang.Float f ? f.FloatValue() : 1f;
        }
        catch
        {
            return 1f;
        }
    }

    /// <summary>
    /// 对焦框覆盖视图，绘制四角 L 形对焦框并带缩放渐隐动画
    /// </summary>
    private class FocusOverlayView : global::Android.Views.View
    {
        private static readonly global::Android.Graphics.Paint FocusPaint;

        static FocusOverlayView()
        {
            FocusPaint = new global::Android.Graphics.Paint
            {
                Color = global::Android.Graphics.Color.ParseColor("#FFE600"),
                StrokeWidth = 3f,
                AntiAlias = true
            };
            FocusPaint.SetStyle(global::Android.Graphics.Paint.Style.Stroke);
        }

        private const int CornerLength = 30;
        private const int FocusSize = 100;
        private const long AnimDuration = 800;

        private float _focusX;
        private float _focusY;
        private float _currentScale = 1.5f;
        private int _currentAlpha = 255;
        private ValueAnimator? _animator;

        public FocusOverlayView(Context context) : base(context)
        {
            SetLayerType(LayerType.Software, null);
        }

        public void ShowFocus(float x, float y)
        {
            _focusX = x;
            _focusY = y;

            _animator?.Cancel();

            var scaleAnim = ValueAnimator.OfFloat(1.5f, 1f);
            scaleAnim.SetDuration(AnimDuration);
            scaleAnim.Update += (_, args) =>
            {
                _currentScale = (float)args.Animation!.AnimatedValue!;
                Invalidate();
            };

            var alphaAnim = ValueAnimator.OfInt(255, 0);
            alphaAnim.SetDuration(AnimDuration);
            alphaAnim.Update += (_, args) =>
            {
                _currentAlpha = (int)args.Animation!.AnimatedValue!;
                Invalidate();
            };

            var set = new AnimatorSet();
            set.PlayTogether(scaleAnim, alphaAnim);
            set.Start();

            _animator = scaleAnim;
            Invalidate();
        }

        protected override void OnDraw(Canvas? canvas)
        {
            base.OnDraw(canvas);
            if (canvas == null) return;

            FocusPaint.Alpha = _currentAlpha;

            var density = Context.Resources!.DisplayMetrics!.Density;
            var size = FocusSize * density * _currentScale;
            var corner = CornerLength * density * _currentScale;

            var left = _focusX - size / 2;
            var top = _focusY - size / 2;
            var right = _focusX + size / 2;
            var bottom = _focusY + size / 2;

            canvas.DrawLine(left, top, left + corner, top, FocusPaint);
            canvas.DrawLine(left, top, left, top + corner, FocusPaint);
            canvas.DrawLine(right, top, right - corner, top, FocusPaint);
            canvas.DrawLine(right, top, right, top + corner, FocusPaint);
            canvas.DrawLine(left, bottom, left + corner, bottom, FocusPaint);
            canvas.DrawLine(left, bottom, left, bottom - corner, FocusPaint);
            canvas.DrawLine(right, bottom, right - corner, bottom, FocusPaint);
            canvas.DrawLine(right, bottom, right, bottom - corner, FocusPaint);
        }
    }

    #region Callbacks

    private class SurfaceTextureListener : Java.Lang.Object, TextureView.ISurfaceTextureListener
    {
        private readonly CameraPreviewHandler _handler;

        public SurfaceTextureListener(CameraPreviewHandler handler) => _handler = handler;

        public void OnSurfaceTextureAvailable(global::Android.Graphics.SurfaceTexture surface, int width, int height)
        {
        }

        public bool OnSurfaceTextureDestroyed(global::Android.Graphics.SurfaceTexture surface) => true;

        public void OnSurfaceTextureSizeChanged(global::Android.Graphics.SurfaceTexture surface, int width, int height)
        {
        }

        public void OnSurfaceTextureUpdated(global::Android.Graphics.SurfaceTexture surface) { }
    }

    private class PreviewSurfaceProvider : Java.Lang.Object, Preview.ISurfaceProvider
    {
        private readonly CameraPreviewHandler _handler;

        public PreviewSurfaceProvider(CameraPreviewHandler handler) => _handler = handler;

        public void OnSurfaceRequested(SurfaceRequest? request)
        {
            if (request == null) return;

            var texture = _handler._textureView?.SurfaceTexture;
            if (texture == null) return;

            try
            {
                var surface = new global::Android.Views.Surface(texture);
                var executor = _handler.Context!.MainExecutor!;
                // ProvideSurface 需要 Surface, Executor, 和 resultListener
                request.ProvideSurface(surface, executor, new SurfaceResultListener());
            }
            catch (System.Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[CameraX] SurfaceProvider 失败: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// SurfaceRequest result listener
    /// </summary>
    private class SurfaceResultListener : Java.Lang.Object, AndroidX.Core.Util.IConsumer
    {
        public void Accept(Java.Lang.Object? result)
        {
            if (result != null)
            {
                System.Diagnostics.Debug.WriteLine($"[CameraX] Surface 结果: {result}");
            }
        }
    }

    private class ImageCaptureCallback : ImageCapture.OnImageCapturedCallback
    {
        private readonly CameraPreviewHandler _handler;

        public ImageCaptureCallback(CameraPreviewHandler handler) => _handler = handler;

        public override void OnCaptureSuccess(IImageProxy? imageProxy)
        {
            try
            {
                if (imageProxy == null) return;

                var buffer = imageProxy.GetPlanes()![0].Buffer!;
                var bytes = new byte[buffer.Remaining()];
                buffer.Get(bytes);

                System.Diagnostics.Debug.WriteLine($"[CameraX] 图像捕获成功: {bytes.Length} 字节");
                _handler.VirtualView?.OnPhotoCaptured(bytes);
            }
            catch (System.Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[CameraX] 读取图像失败: {ex.Message}");
            }
            finally
            {
                imageProxy?.Close();
            }
        }

        public override void OnError(ImageCaptureException error)
        {
            System.Diagnostics.Debug.WriteLine($"[CameraX] 拍照错误: {error.Message}");
            _handler.VirtualView?.OnError($"拍照错误: {error.Message}");
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

    /// <summary>
    /// 设备物理方向监听器
    /// </summary>
    private class DeviceOrientationListener : OrientationEventListener
    {
        private readonly CameraPreviewHandler _handler;

        public DeviceOrientationListener(Context context, CameraPreviewHandler handler)
            : base(context, global::Android.Hardware.SensorDelay.Normal)
        {
            _handler = handler;
        }

        public override void OnOrientationChanged(int orientation)
        {
            _handler.OnOrientationChanged(orientation);
        }
    }

    /// <summary>
    /// 单击手势监听器，触发对焦和测光
    /// </summary>
    private class TapListener : Java.Lang.Object, GestureDetector.IOnGestureListener
    {
        private readonly CameraPreviewHandler _handler;

        public TapListener(CameraPreviewHandler handler) => _handler = handler;

        public bool OnDown(MotionEvent e) => true;

        public bool OnFling(MotionEvent? e1, MotionEvent e2, float velocityX, float velocityY) => false;

        public void OnLongPress(MotionEvent e) { }

        public bool OnScroll(MotionEvent? e1, MotionEvent e2, float distanceX, float distanceY) => false;

        public void OnShowPress(MotionEvent e) { }

        public bool OnSingleTapUp(MotionEvent e)
        {
            _handler.FocusAndMeter(e.GetX(), e.GetY());
            _handler._focusOverlay?.ShowFocus(e.GetX(), e.GetY());
            return true;
        }
    }

    #endregion
}
