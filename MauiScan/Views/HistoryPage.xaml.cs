using MauiScan.Models;
using MauiScan.Services;
using MauiScan.Services.Sync;
using System.Collections.ObjectModel;
using System.Windows.Input;

namespace MauiScan.Views;

public partial class HistoryPage : ContentPage
{
    private readonly ScanSyncService _syncService;
    private readonly IClipboardService _clipboardService;

    public ObservableCollection<HistoryItemViewModel> HistoryItems { get; } = new();
    public bool IsRefreshing { get; set; }
    public ICommand RefreshCommand { get; }

    public HistoryPage(ScanSyncService syncService, IClipboardService clipboardService)
    {
        InitializeComponent();

        _syncService = syncService;
        _clipboardService = clipboardService;

        RefreshCommand = new Command(async () => await LoadHistoryAsync());

        BindingContext = this;
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        await LoadHistoryAsync();
    }

    private async Task LoadHistoryAsync()
    {
        try
        {
            LoadingIndicator.IsRunning = true;
            LoadingIndicator.IsVisible = true;

            var scans = await _syncService.GetRecentScansAsync(20);

            HistoryItems.Clear();
            foreach (var scan in scans)
            {
                HistoryItems.Add(new HistoryItemViewModel(scan));
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"加载历史记录失败: {ex.Message}", "确定");
        }
        finally
        {
            LoadingIndicator.IsRunning = false;
            LoadingIndicator.IsVisible = false;
            IsRefreshing = false;
        }
    }

    private async void OnRefreshClicked(object sender, EventArgs e)
    {
        await LoadHistoryAsync();
    }

    private async void OnItemSelected(object? sender, SelectionChangedEventArgs e)
    {
        if (e.CurrentSelection.FirstOrDefault() is not HistoryItemViewModel selectedItem)
            return;

        // 清除选中状态
        HistoryCollectionView.SelectedItem = null;

        // 显示详情
        await ShowDetailAsync(selectedItem);
    }

    private async Task ShowDetailAsync(HistoryItemViewModel item)
    {
        try
        {
            LoadingIndicator.IsRunning = true;
            LoadingIndicator.IsVisible = true;

            // 下载图片
            var imageData = await _syncService.DownloadScanAsync(item.ScanImage.DownloadUrl);
            if (imageData == null)
            {
                await DisplayAlert("错误", "下载图片失败", "确定");
                return;
            }

            // 显示操作选项
            var action = await DisplayActionSheet(
                $"{item.ScannedAtText}\n{item.SizeText}",
                "取消",
                null,
                "复制到剪贴板",
                "保存到相册");

            switch (action)
            {
                case "复制到剪贴板":
                    await _clipboardService.CopyImageToClipboardAsync(imageData);
                    await DisplayAlert("成功", "已复制到剪贴板", "确定");
                    break;

                case "保存到相册":
                    await SaveToGalleryAsync(imageData, item.ScanImage.FileName);
                    break;
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"操作失败: {ex.Message}", "确定");
        }
        finally
        {
            LoadingIndicator.IsRunning = false;
            LoadingIndicator.IsVisible = false;
        }
    }

    private async Task SaveToGalleryAsync(byte[] imageData, string fileName)
    {
        try
        {
#if ANDROID
            var context = Android.App.Application.Context;
            var contentResolver = context.ContentResolver;

            var contentValues = new Android.Content.ContentValues();
            contentValues.Put(Android.Provider.MediaStore.IMediaColumns.DisplayName, fileName);
            contentValues.Put(Android.Provider.MediaStore.IMediaColumns.MimeType, "image/jpeg");

            if (Android.OS.Build.VERSION.SdkInt >= Android.OS.BuildVersionCodes.Q)
            {
                contentValues.Put(Android.Provider.MediaStore.IMediaColumns.RelativePath,
                    Android.OS.Environment.DirectoryPictures + "/MauiScan");
            }

            var uri = contentResolver?.Insert(
                Android.Provider.MediaStore.Images.Media.ExternalContentUri,
                contentValues);

            if (uri != null)
            {
                using var outputStream = contentResolver?.OpenOutputStream(uri);
                if (outputStream != null)
                {
                    await outputStream.WriteAsync(imageData);
                    await DisplayAlert("成功", "已保存到相册", "确定");
                }
            }
#elif IOS || MACCATALYST
            var image = UIKit.UIImage.LoadFromData(Foundation.NSData.FromArray(imageData));
            if (image != null)
            {
                image.SaveToPhotosAlbum((img, error) =>
                {
                    MainThread.BeginInvokeOnMainThread(async () =>
                    {
                        if (error == null)
                        {
                            await DisplayAlert("成功", "已保存到相册", "确定");
                        }
                        else
                        {
                            await DisplayAlert("错误", $"保存失败: {error.LocalizedDescription}", "确定");
                        }
                    });
                });
            }
#endif
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"保存失败: {ex.Message}", "确定");
        }
    }
}

public class HistoryItemViewModel
{
    public ScanImageDto ScanImage { get; }
    public ImageSource? ThumbnailSource { get; set; }
    public string ScannedAtText { get; }
    public string SizeText { get; }

    public HistoryItemViewModel(ScanImageDto scanImage)
    {
        ScanImage = scanImage;
        ScannedAtText = scanImage.ScannedAt.ToString("yyyy-MM-dd HH:mm:ss");
        SizeText = $"{scanImage.Width} × {scanImage.Height}";

        // 异步加载缩略图
        LoadThumbnailAsync();
    }

    private async void LoadThumbnailAsync()
    {
        try
        {
            var httpClient = new HttpClient();
            var imageData = await httpClient.GetByteArrayAsync(ScanImage.DownloadUrl);
            ThumbnailSource = ImageSource.FromStream(() => new MemoryStream(imageData));
        }
        catch
        {
            // 加载失败使用占位图
        }
    }
}
