using AVFoundation;
using Foundation;
using MauiScan.Services;
using Photos;
using UIKit;

namespace MauiScan.Platforms.iOS.Services
{
    public class CameraService : ICameraService
    {
        public Action<string>? OnDebugLog { get; set; }
        private TaskCompletionSource<byte[]?>? _photoTcs;

        private void Log(string message)
        {
            Console.WriteLine($"[CameraService] {message}");
            OnDebugLog?.Invoke($"[CameraService] {message}");
        }

        public async Task<byte[]?> TakePhotoAsync()
        {
            try
            {
                Log("TakePhotoAsync called - using UIImagePickerController");

                _photoTcs = new TaskCompletionSource<byte[]?>();

                // 在主线程上执行
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    try
                    {
                        Log("Checking camera availability...");
                        if (!UIImagePickerController.IsSourceTypeAvailable(UIImagePickerControllerSourceType.Camera))
                        {
                            Log("Camera not available on this device");
                            _photoTcs?.TrySetResult(null);
                            return;
                        }

                        Log("Creating picker...");
                        var picker = new UIImagePickerController();
                        picker.SourceType = UIImagePickerControllerSourceType.Camera;
                        picker.AllowsEditing = false;

                        picker.FinishedPickingMedia += (s, e) =>
                        {
                            Log("FinishedPickingMedia");
                            try
                            {
                                var image = e.OriginalImage;
                                if (image != null)
                                {
                                    using var data = image.AsJPEG(0.9f);
                                    if (data != null)
                                    {
                                        var bytes = new byte[data.Length];
                                        System.Runtime.InteropServices.Marshal.Copy(data.Bytes, bytes, 0, (int)data.Length);
                                        Log($"Got {bytes.Length} bytes");
                                        _photoTcs?.TrySetResult(bytes);
                                    }
                                    else
                                    {
                                        _photoTcs?.TrySetResult(null);
                                    }
                                }
                                else
                                {
                                    _photoTcs?.TrySetResult(null);
                                }
                            }
                            catch (Exception ex)
                            {
                                Log($"Error processing image: {ex.Message}");
                                _photoTcs?.TrySetResult(null);
                            }
                            picker.DismissViewController(true, null);
                        };

                        picker.Canceled += (s, e) =>
                        {
                            Log("User cancelled");
                            _photoTcs?.TrySetResult(null);
                            picker.DismissViewController(true, null);
                        };

                        Log("Getting view controller...");
                        var vc = Platform.GetCurrentUIViewController();
                        if (vc == null)
                        {
                            Log("No view controller found");
                            _photoTcs?.TrySetResult(null);
                            return;
                        }

                        Log($"Presenting picker from {vc.GetType().Name}...");
                        vc.PresentViewController(picker, true, () =>
                        {
                            Log("Picker presented");
                        });
                    }
                    catch (Exception ex)
                    {
                        Log($"Error in MainThread: {ex.GetType().Name}: {ex.Message}");
                        _photoTcs?.TrySetResult(null);
                    }
                });

                Log("Waiting for result...");
                return await _photoTcs.Task;
            }
            catch (Exception ex)
            {
                Log($"TakePhotoAsync exception: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        public async Task<bool> CheckPermissionsAsync()
        {
            // 简化实现：iOS 会自动在 UIImagePickerController 中处理权限
            return true;
        }

        public async Task<bool> RequestPermissionsAsync()
        {
            // 简化实现：iOS 会自动在 UIImagePickerController 中处理权限
            return true;
        }
    }
}
