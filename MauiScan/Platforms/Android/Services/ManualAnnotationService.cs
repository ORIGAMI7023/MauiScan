using Android.Content;
using MauiScan.Services;

namespace MauiScan.Platforms.Android.Services;

public class ManualAnnotationService : IManualAnnotationService
{
    private static TaskCompletionSource<ManualAnnotationResult?>? _tcs;
    private static byte[]? _currentImageBytes;

    public Task<ManualAnnotationResult?> AnnotateAsync(byte[] imageBytes)
    {
        _tcs = new TaskCompletionSource<ManualAnnotationResult?>();
        _currentImageBytes = imageBytes;

        // 启动 Android Activity
        var context = Platform.CurrentActivity;
        if (context == null)
        {
            _tcs.SetResult(null);
            return _tcs.Task;
        }

        var intent = new Intent(context, typeof(ManualAnnotationActivity));
        context.StartActivity(intent);

        return _tcs.Task;
    }

    public static byte[]? GetCurrentImageBytes() => _currentImageBytes;

    public static void SetResult(ManualAnnotationResult? result)
    {
        _tcs?.TrySetResult(result);
        _tcs = null;
        _currentImageBytes = null;
    }
}
