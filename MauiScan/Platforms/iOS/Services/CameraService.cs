using Foundation;
using MauiScan.Services;
using Photos;
using UIKit;

namespace MauiScan.Platforms.iOS.Services
{
    public class CameraService : ICameraService
    {
        public async Task<byte[]?> TakePhotoAsync()
        {
            try
            {
                var tcs = new TaskCompletionSource<byte[]?>();

                await MainThread.InvokeOnMainThreadAsync(async () =>
                {
                    try
                    {
                        // 检查相机可用性
                        if (!UIImagePickerController.IsSourceTypeAvailable(UIImagePickerControllerSourceType.Camera))
                        {
                            tcs.SetResult(null);
                            return;
                        }

                        // 创建图片选择器
                        var picker = new UIImagePickerController
                        {
                            SourceType = UIImagePickerControllerSourceType.Camera,
                            CameraCaptureMode = UIImagePickerControllerCameraCaptureMode.Photo,
                            AllowsEditing = false
                        };

                        // 设置完成回调
                        picker.FinishedPickingMedia += (sender, e) =>
                        {
                            try
                            {
                                var image = e.OriginalImage;
                                if (image != null)
                                {
                                    // 转换为 JPEG 数据
                                    using var imageData = image.AsJPEG(0.9f);
                                    if (imageData != null)
                                    {
                                        var bytes = new byte[imageData.Length];
                                        System.Runtime.InteropServices.Marshal.Copy(imageData.Bytes, bytes, 0, Convert.ToInt32(imageData.Length));
                                        tcs.SetResult(bytes);
                                    }
                                    else
                                    {
                                        tcs.SetResult(null);
                                    }
                                }
                                else
                                {
                                    tcs.SetResult(null);
                                }

                                picker.DismissViewController(true, null);
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error in FinishedPickingMedia: {ex}");
                                tcs.SetException(ex);
                                picker.DismissViewController(true, null);
                            }
                        };

                        // 设置取消回调
                        picker.Canceled += (sender, e) =>
                        {
                            tcs.SetResult(null);
                            picker.DismissViewController(true, null);
                        };

                        // 显示相机界面
                        var viewController = Platform.GetCurrentUIViewController();
                        if (viewController != null)
                        {
                            await viewController.PresentViewControllerAsync(picker, true);
                        }
                        else
                        {
                            tcs.SetResult(null);
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error in TakePhotoAsync: {ex}");
                        tcs.SetException(ex);
                    }
                });

                return await tcs.Task;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"TakePhotoAsync exception: {ex}");
                return null;
            }
        }

        public async Task<bool> CheckPermissionsAsync()
        {
            try
            {
                var status = await Permissions.CheckStatusAsync<Permissions.Camera>();
                return status == PermissionStatus.Granted;
            }
            catch
            {
                return false;
            }
        }

        public async Task<bool> RequestPermissionsAsync()
        {
            try
            {
                var status = await Permissions.RequestAsync<Permissions.Camera>();
                return status == PermissionStatus.Granted;
            }
            catch
            {
                return false;
            }
        }
    }
}
