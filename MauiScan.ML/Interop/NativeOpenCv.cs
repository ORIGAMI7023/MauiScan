using System.Runtime.InteropServices;

namespace MauiScan.ML.Interop;

/// <summary>
/// Native OpenCV 库 P/Invoke 互操作
/// </summary>
internal static class NativeOpenCv
{
#if ANDROID
    private const string LibraryName = "opencv_scanner";
#elif IOS || MACCATALYST
    private const string LibraryName = "@rpath/opencv_scanner.framework/opencv_scanner";
#else
    private const string LibraryName = "opencv_scanner"; // 不应该被调用
#endif

    /// <summary>
    /// 精修单个角点
    /// </summary>
    /// <param name="inputData">输入图像数据（JPEG/PNG）</param>
    /// <param name="inputSize">输入数据大小</param>
    /// <param name="mlX">ML 预测的 X 坐标</param>
    /// <param name="mlY">ML 预测的 Y 坐标</param>
    /// <param name="refinedX">输出精修后的 X 坐标</param>
    /// <param name="refinedY">输出精修后的 Y 坐标</param>
    /// <returns>1=成功, 0=失败</returns>
    [DllImport(LibraryName, EntryPoint = "scanner_refine_corner", CallingConvention = CallingConvention.Cdecl)]
    public static extern int RefineCorner(
        byte[] inputData,
        int inputSize,
        float mlX,
        float mlY,
        out float refinedX,
        out float refinedY
    );

    /// <summary>
    /// 获取库版本信息
    /// </summary>
    [DllImport(LibraryName, EntryPoint = "scanner_get_version", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr GetVersion();
}
