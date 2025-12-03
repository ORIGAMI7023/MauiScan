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
    private static extern IntPtr scanner_get_version();

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
                var parameters = scanner_get_default_params();
                var nativeResult = new NativeScanResult
                {
                    ErrorMessage = new byte[256]
                };

                int returnCode = scanner_process_scan(
                    imageBytes,
                    imageBytes.Length,
                    applyEnhancement ? 1 : 0,
                    ref parameters,
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
                return ScanResult.Failure($"Native 调用失败: {ex.Message}");
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
}
