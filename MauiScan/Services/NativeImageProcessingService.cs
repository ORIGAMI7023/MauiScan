using MauiScan.Models;
using System.Runtime.InteropServices;

namespace MauiScan.Services;

/// <summary>
/// 基于 Native C++ OpenCV 的图像处理服务实现
/// 通过 P/Invoke 调用原生 C++ 库
/// </summary>
public class NativeImageProcessingService : IImageProcessingService
{
    // P/Invoke 结构体定义

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeQuadPoints
    {
        public int TopLeftX;
        public int TopLeftY;
        public int TopRightX;
        public int TopRightY;
        public int BottomRightX;
        public int BottomRightY;
        public int BottomLeftX;
        public int BottomLeftY;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeScanResult
    {
        public IntPtr ImageData;
        public int ImageSize;
        public int Width;
        public int Height;
        public NativeQuadPoints Quad;
        public int Success;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
        public byte[] ErrorMessage;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeScannerParams
    {
        public double CannyThreshold1;
        public double CannyThreshold2;
        public int GaussianKernelSize;
        public double MinContourAreaRatio;
        public int JpegQuality;
    }

    // P/Invoke 函数声明

#if ANDROID
    private const string LibraryName = "opencv_scanner";
#elif IOS
    private const string LibraryName = "__Internal";
#else
    private const string LibraryName = "opencv_scanner";
#endif

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern NativeScannerParams scanner_get_default_params();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int scanner_process_scan(
        byte[] inputData,
        int inputSize,
        int applyEnhancement,
        ref NativeScannerParams parameters,
        ref NativeScanResult result
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int scanner_detect_bounds(
        byte[] inputData,
        int inputSize,
        ref NativeScannerParams parameters,
        ref NativeQuadPoints quad
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void scanner_free_result(ref NativeScanResult result);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int scanner_apply_transform(
        byte[] inputData,
        int inputSize,
        ref NativeQuadPoints quad,
        int applyEnhancement,
        int jpegQuality,
        ref NativeScanResult result
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr scanner_get_version();

    // 公开的 QuadPoints 类型（用于手动标注）
    public class QuadPoints
    {
        public float TopLeftX { get; set; }
        public float TopLeftY { get; set; }
        public float TopRightX { get; set; }
        public float TopRightY { get; set; }
        public float BottomRightX { get; set; }
        public float BottomRightY { get; set; }
        public float BottomLeftX { get; set; }
        public float BottomLeftY { get; set; }
    }

    // 辅助方法

    private static QuadrilateralPoints ConvertToManagedQuad(NativeQuadPoints nativeQuad)
    {
        return new QuadrilateralPoints(
            new Point2D(nativeQuad.TopLeftX, nativeQuad.TopLeftY),
            new Point2D(nativeQuad.TopRightX, nativeQuad.TopRightY),
            new Point2D(nativeQuad.BottomRightX, nativeQuad.BottomRightY),
            new Point2D(nativeQuad.BottomLeftX, nativeQuad.BottomLeftY)
        );
    }

    private static string GetErrorMessage(byte[] errorBytes)
    {
        int nullIndex = Array.IndexOf(errorBytes, (byte)0);
        if (nullIndex >= 0)
        {
            return System.Text.Encoding.UTF8.GetString(errorBytes, 0, nullIndex);
        }
        return System.Text.Encoding.UTF8.GetString(errorBytes);
    }

    // IImageProcessingService 接口实现

    public async Task<ScanResult> ProcessScanAsync(byte[] imageBytes, bool applyEnhancement = false)
    {
        return await Task.Run(() =>
        {
            try
            {
                System.Diagnostics.Debug.WriteLine($"[Native] ====== 新的扫描请求 ======");
                System.Diagnostics.Debug.WriteLine($"[Native] 图像大小: {imageBytes.Length} 字节, 前4字节: {imageBytes[0]:X2} {imageBytes[1]:X2} {imageBytes[2]:X2} {imageBytes[3]:X2}");

                // 测试库是否可加载
                try
                {
                    var version = GetVersion();
                    System.Diagnostics.Debug.WriteLine($"[Native] OpenCV Scanner 版本: {version}");
                }
                catch (Exception vex)
                {
                    System.Diagnostics.Debug.WriteLine($"[Native] 无法获取版本: {vex.GetType().Name} - {vex.Message}");
                    return ScanResult.Failure($"无法加载 Native 库: {vex.Message}");
                }

                var parameters = scanner_get_default_params();
                System.Diagnostics.Debug.WriteLine($"[Native] 参数: Canny={parameters.CannyThreshold1},{parameters.CannyThreshold2}, Gaussian={parameters.GaussianKernelSize}, MinArea={parameters.MinContourAreaRatio}, Quality={parameters.JpegQuality}");

                // 检查参数是否被污染
                if (parameters.CannyThreshold1 != 50 || parameters.CannyThreshold2 != 150 ||
                    parameters.GaussianKernelSize != 5 || parameters.JpegQuality != 95 ||
                    parameters.MinContourAreaRatio < 0.05 || parameters.MinContourAreaRatio > 0.5)
                {
                    System.Diagnostics.Debug.WriteLine($"[Native] ⚠️ 参数异常！重新获取默认参数");
                    parameters = scanner_get_default_params();
                }

                var nativeResult = new NativeScanResult
                {
                    ImageData = IntPtr.Zero,
                    ImageSize = 0,
                    Width = 0,
                    Height = 0,
                    Success = 0,
                    ErrorMessage = new byte[256]
                };

                int returnCode = scanner_process_scan(
                    imageBytes,
                    imageBytes.Length,
                    applyEnhancement ? 1 : 0,
                    ref parameters,
                    ref nativeResult
                );

                System.Diagnostics.Debug.WriteLine($"[Native] 返回码: {returnCode}, Success: {nativeResult.Success}");

                if (returnCode != 0 || nativeResult.Success == 0)
                {
                    string errorMsg = GetErrorMessage(nativeResult.ErrorMessage);
                    System.Diagnostics.Debug.WriteLine($"[Native] 错误: {errorMsg}");
                    return ScanResult.Failure(errorMsg);
                }

                // 从非托管内存复制图像数据
                byte[] resultImageData = new byte[nativeResult.ImageSize];
                Marshal.Copy(nativeResult.ImageData, resultImageData, 0, nativeResult.ImageSize);

                // 释放 Native 内存
                scanner_free_result(ref nativeResult);

                var quad = ConvertToManagedQuad(nativeResult.Quad);

                System.Diagnostics.Debug.WriteLine($"[Native] 成功处理，输出大小: {nativeResult.ImageSize} 字节");

                return new ScanResult(
                    resultImageData,
                    nativeResult.Width,
                    nativeResult.Height,
                    quad
                );
            }
            catch (DllNotFoundException dex)
            {
                System.Diagnostics.Debug.WriteLine($"[Native] DllNotFoundException: {dex.Message}");
                return ScanResult.Failure($"找不到 Native 库 libopencv_scanner.so");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[Native] Exception: {ex.GetType().Name} - {ex.Message}");
                System.Diagnostics.Debug.WriteLine($"[Native] StackTrace: {ex.StackTrace}");
                return ScanResult.Failure($"Native 调用失败: {ex.GetType().Name} - {ex.Message}");
            }
        });
    }

    public async Task<QuadrilateralPoints?> DetectDocumentBoundsAsync(byte[] imageBytes)
    {
        return await Task.Run(() =>
        {
            try
            {
                var parameters = scanner_get_default_params();
                var nativeQuad = new NativeQuadPoints();

                int detected = scanner_detect_bounds(
                    imageBytes,
                    imageBytes.Length,
                    ref parameters,
                    ref nativeQuad
                );

                if (detected == 0)
                {
                    return null;
                }

                return ConvertToManagedQuad(nativeQuad);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"边界检测失败: {ex.Message}");
                return null;
            }
        });
    }

    public string GetVersion()
    {
        try
        {
            IntPtr versionPtr = scanner_get_version();
            return Marshal.PtrToStringAnsi(versionPtr) ?? "Unknown";
        }
        catch
        {
            return "Unknown";
        }
    }

    /// <summary>
    /// 对图片应用透视变换（用于手动标注）
    /// </summary>
    public ScanResult ApplyPerspectiveTransform(byte[] imageBytes, QuadPoints userQuad, bool applyEnhancement = false)
    {
        try
        {
            // 转换用户坐标到 Native 坐标
            var nativeQuad = new NativeQuadPoints
            {
                TopLeftX = (int)userQuad.TopLeftX,
                TopLeftY = (int)userQuad.TopLeftY,
                TopRightX = (int)userQuad.TopRightX,
                TopRightY = (int)userQuad.TopRightY,
                BottomRightX = (int)userQuad.BottomRightX,
                BottomRightY = (int)userQuad.BottomRightY,
                BottomLeftX = (int)userQuad.BottomLeftX,
                BottomLeftY = (int)userQuad.BottomLeftY
            };

            var nativeResult = new NativeScanResult
            {
                ImageData = IntPtr.Zero,
                ImageSize = 0,
                Width = 0,
                Height = 0,
                Success = 0,
                ErrorMessage = new byte[256]
            };

            int returnCode = scanner_apply_transform(
                imageBytes,
                imageBytes.Length,
                ref nativeQuad,
                applyEnhancement ? 1 : 0,
                95, // JPEG quality
                ref nativeResult
            );

            if (returnCode != 0 || nativeResult.Success == 0)
            {
                string errorMsg = GetErrorMessage(nativeResult.ErrorMessage);
                return ScanResult.Failure(errorMsg);
            }

            // 从非托管内存复制图像数据
            byte[] resultImageData = new byte[nativeResult.ImageSize];
            Marshal.Copy(nativeResult.ImageData, resultImageData, 0, nativeResult.ImageSize);

            // 释放 Native 内存
            scanner_free_result(ref nativeResult);

            var quad = ConvertToManagedQuad(nativeResult.Quad);

            return new ScanResult(
                resultImageData,
                nativeResult.Width,
                nativeResult.Height,
                quad
            );
        }
        catch (Exception ex)
        {
            return ScanResult.Failure($"透视变换失败: {ex.Message}");
        }
    }
}
