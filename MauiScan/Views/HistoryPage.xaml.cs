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
    private readonly IConfigService _configService;

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

    public HistoryPage(ScanSyncService syncService, IClipboardService clipboardService, IConfigService configService)
    {
        InitializeComponent();

        _syncService = syncService;
        _clipboardService = clipboardService;
        _configService = configService;

        RefreshCommand = new Command(async () => await LoadHistoryAsync());
        DeleteCommand = new Command<HistoryItemViewModel>(async (item) => await DeleteItemAsync(item));

        // 监听连接状态变化
        _syncService.ConnectionStateChanged += OnConnectionStateChanged;

        // 监听错误事件
        _syncService.ErrorOccurred += OnErrorOccurred;

        BindingContext = this;
    }

    internal ScanSyncService SyncService => _syncService;

    protected override async void OnAppearing()
    {
        base.OnAppearing();

        // 初始化配置并连接服务器
        try
        {
            var config = await _configService.LoadConfigAsync();
            System.Diagnostics.Debug.WriteLine($"配置加载成功: ServerUrl={config.ServerUrl}, ApiKey={(string.IsNullOrEmpty(config.ApiKey) ? "未设置" : "已设置")}");

            // 如果未连接，尝试连接
            if (!_syncService.IsConnected)
            {
                await _syncService.ConnectAsync();
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"配置加载失败: {ex.Message}");
        }

        // 主动同步连接状态到 UI（避免已连接时事件不触发导致状态显示错误）
        OnConnectionStateChanged(_syncService.IsConnected);

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

            // 增量更新：删除不存在的，添加新增的，保持顺序
            var newFileNames = scans.Select(s => s.FileName).ToHashSet();
            var toRemove = HistoryItems.Where(i => !newFileNames.Contains(i.ScanImage.FileName)).ToList();
            foreach (var item in toRemove)
                HistoryItems.Remove(item);

            var existingFileNames = HistoryItems.Select(i => i.ScanImage.FileName).ToHashSet();
            for (int i = 0; i < scans.Count; i++)
            {
                var scan = scans[i];
                if (!existingFileNames.Contains(scan.FileName))
                {
                    HistoryItems.Insert(i, new HistoryItemViewModel(scan, _syncService));
                }
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
            // 先弹窗，不需要等待下载
            var action = await DisplayActionSheet(
                $"{item.ScannedAtText}\n{item.SizeText}",
                "删除",
                "取消",
                null,
                "复制到剪贴板",
                "保存到相册");

            switch (action)
            {
                case "复制到剪贴板":
                case "保存到相册":
                    // 只有需要图片数据时才下载
                    LoadingIndicator.IsRunning = true;
                    LoadingIndicator.IsVisible = true;
                    var imageData = await _syncService.DownloadScanAsync(item.ScanImage.DownloadUrl);
                    if (imageData == null)
                    {
                        await DisplayAlert("错误", "下载图片失败", "确定");
                        return;
                    }
                    if (action == "复制到剪贴板")
                    {
                        await _clipboardService.CopyImageToClipboardAsync(imageData);
                        await DisplayAlert("成功", "已复制到剪贴板", "确定");
                    }
                    else
                    {
                        await SaveToGalleryAsync(imageData, item.ScanImage.FileName);
                    }
                    break;

                case "删除":
                    await DeleteItemAsync(item);
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

    private void OnConnectionStateChanged(bool isConnected)
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            if (isConnected)
            {
                ConnectionStatusDot.Fill = new SolidColorBrush(Colors.Green);
                ConnectionStatusLabel.Text = "已连接";
            }
            else
            {
                ConnectionStatusDot.Fill = new SolidColorBrush(Colors.Red);
                ConnectionStatusLabel.Text = "未连接";
            }
        });
    }

    private async void OnErrorOccurred(string errorMessage)
    {
        await MainThread.InvokeOnMainThreadAsync(async () =>
        {
            await DisplayAlert("同步错误", errorMessage, "确定");
        });
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
            // 优先使用缩略图，回退到完整图片
            var url = !string.IsNullOrEmpty(ScanImage.ThumbnailUrl) ? ScanImage.ThumbnailUrl : ScanImage.DownloadUrl;
            var imageData = await _syncService.DownloadScanAsync(url);
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
