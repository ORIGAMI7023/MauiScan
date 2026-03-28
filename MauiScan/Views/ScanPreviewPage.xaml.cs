namespace MauiScan.Views;

public partial class ScanPreviewPage : ContentPage
{
    private byte[] _imageData;

    /// <summary>
    /// 用户点击"使用"时触发，参数为最终图片数据
    /// </summary>
    public event Action<byte[]>? Confirmed;

    /// <summary>
    /// 用户点击"重拍"时触发
    /// </summary>
    public event Action? Retake;

    public ScanPreviewPage(byte[] imageData)
    {
        InitializeComponent();
        _imageData = imageData;
        PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(imageData));
    }

    private async void OnRotateClicked(object sender, EventArgs e)
    {
        RotateButton.IsEnabled = false;
        ConfirmButton.IsEnabled = false;
        try
        {
            var rotated = await Task.Run(() => RotateJpeg90(_imageData));
            if (rotated != null)
            {
                _imageData = rotated;
                PreviewImage.Source = ImageSource.FromStream(() => new MemoryStream(rotated));
            }
        }
        finally
        {
            RotateButton.IsEnabled = true;
            ConfirmButton.IsEnabled = true;
        }
    }

    private async void OnConfirmClicked(object sender, EventArgs e)
    {
        var data = _imageData;
        await Navigation.PopAsync();
        Confirmed?.Invoke(data);
    }

    private async void OnRetakeClicked(object sender, EventArgs e)
    {
        await Navigation.PopAsync();
        Retake?.Invoke();
    }

    private byte[]? RotateJpeg90(byte[] imageBytes)
    {
#if ANDROID
        using var bitmap = Android.Graphics.BitmapFactory.DecodeByteArray(imageBytes, 0, imageBytes.Length);
        if (bitmap == null) return null;

        var matrix = new Android.Graphics.Matrix();
        matrix.PostRotate(90);

        using var rotated = Android.Graphics.Bitmap.CreateBitmap(
            bitmap, 0, 0, bitmap.Width, bitmap.Height, matrix, true);

        using var stream = new MemoryStream();
        rotated.Compress(Android.Graphics.Bitmap.CompressFormat.Jpeg, 90, stream);
        return stream.ToArray();
#else
        return imageBytes;
#endif
    }
}
