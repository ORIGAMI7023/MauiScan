using MauiScan.Models;
using MauiScan.Services;
using MauiScan.Services.Sync;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;

namespace MauiScan.Views;

public partial class HistoryPage : ContentPage, INotifyPropertyChanged
{
    private readonly ScanSyncService _syncService;
    private readonly IClipboardService _clipboardService;

    public ObservableCollection<HistoryItemViewModel> HistoryItems { get; } = new();

    private bool _isRefreshing;
    public bool IsRefreshing
    {
        get => _isRefreshing;
        set
        {
            _isRefreshing = value;
            OnPropertyChanged();
        }
    }

    public ICommand RefreshCommand { get; }
    public ICommand DeleteCommand { get; }

    public new event PropertyChangedEventHandler? PropertyChanged;

    protected new void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    public HistoryPage(ScanSyncService syncService, IClipboardService clipboardService)
    {
        InitializeComponent();

        _syncService = syncService;
        _clipboardService = clipboardService;

        RefreshCommand = new Command(async () => await LoadHistoryAsync());
        DeleteCommand = new Command<HistoryItemViewModel>(async (item) => await DeleteItemAsync(item));

        BindingContext = this;
    }

    internal ScanSyncService SyncService => _syncService;

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        await LoadHistoryAsync(showLoadingIndicator: true);
    }

    private async Task LoadHistoryAsync(bool showLoadingIndicator = false)
    {
        try
        {
            // 只在首次加载时显示独立的加载指示器
            if (showLoadingIndicator && !IsRefreshing)
            {
                LoadingIndicator.IsRunning = true;
                LoadingIndicator.IsVisible = true;
            }

            var scans = await _syncService.GetRecentScansAsync(20);

            HistoryItems.Clear();
            foreach (var scan in scans)
            {
                HistoryItems.Add(new HistoryItemViewModel(scan, _syncService));
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"加载历史记录失败: {ex.Message}", "确定");
        }
        finally
        {
            // 隐藏加载指示器
            if (LoadingIndicator.IsRunning)
            {
                LoadingIndicator.IsRunning = false;
                LoadingIndicator.IsVisible = false;
            }

            // 停止下拉刷新动画
            IsRefreshing = false;
        }
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

    private async Task DeleteItemAsync(HistoryItemViewModel item)
    {
        var confirm = await DisplayAlert("确认删除",
            $"确定要删除 {item.ScannedAtText} 的扫描记录吗？",
            "删除", "取消");

        if (!confirm) return;

        try
        {
            LoadingIndicator.IsRunning = true;
            LoadingIndicator.IsVisible = true;

            var success = await _syncService.DeleteScanAsync(item.ScanImage.FileName);

            if (success)
            {
                HistoryItems.Remove(item);
            }
            else
            {
                await DisplayAlert("错误", "删除失败", "确定");
            }
        }
        catch (Exception ex)
        {
            await DisplayAlert("错误", $"删除失败: {ex.Message}", "确定");
        }
        finally
        {
            LoadingIndicator.IsRunning = false;
            LoadingIndicator.IsVisible = false;
        }
    }
}

public class HistoryItemViewModel : System.ComponentModel.INotifyPropertyChanged
{
    private readonly ScanSyncService _syncService;

    public ScanImageDto ScanImage { get; }

    private ImageSource? _thumbnailSource;
    public ImageSource? ThumbnailSource
    {
        get => _thumbnailSource;
        set
        {
            _thumbnailSource = value;
            PropertyChanged?.Invoke(this, new System.ComponentModel.PropertyChangedEventArgs(nameof(ThumbnailSource)));
        }
    }

    public string ScannedAtText { get; }
    public string SizeText { get; }

    public event System.ComponentModel.PropertyChangedEventHandler? PropertyChanged;

    public HistoryItemViewModel(ScanImageDto scanImage, ScanSyncService syncService)
    {
        ScanImage = scanImage;
        _syncService = syncService;
        ScannedAtText = scanImage.ScannedAt.ToString("yyyy-MM-dd HH:mm:ss");
        SizeText = $"{scanImage.Width} × {scanImage.Height}";

        // 异步加载缩略图
        LoadThumbnailAsync();
    }

    private async void LoadThumbnailAsync()
    {
        try
        {
            var imageData = await _syncService.DownloadScanAsync(ScanImage.DownloadUrl);
            if (imageData != null)
            {
                ThumbnailSource = ImageSource.FromStream(() => new MemoryStream(imageData));
            }
        }
        catch
        {
            // 加载失败使用占位图
        }
    }
}
