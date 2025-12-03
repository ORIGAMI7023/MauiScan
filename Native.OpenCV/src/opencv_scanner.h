/**
 * opencv_scanner.h - C Interface for OpenCV Document Scanner
 *
 * 纯 C 接口，用于 P/Invoke 跨平台调用
 * 支持 Android (JNI) 和 iOS
 */

#ifndef OPENCV_SCANNER_H
#define OPENCV_SCANNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 导出宏定义
#if defined(_WIN32) || defined(__CYGWIN__)
    #define SCANNER_API __declspec(dllexport)
#elif defined(__ANDROID__) || defined(__APPLE__)
    #define SCANNER_API __attribute__((visibility("default")))
#else
    #define SCANNER_API
#endif

// 四边形顶点结构
typedef struct {
    int32_t top_left_x;
    int32_t top_left_y;
    int32_t top_right_x;
    int32_t top_right_y;
    int32_t bottom_right_x;
    int32_t bottom_right_y;
    int32_t bottom_left_x;
    int32_t bottom_left_y;
} QuadPoints;

// 扫描结果结构
typedef struct {
    uint8_t* image_data;      // JPEG 编码后的图像数据
    int32_t image_size;       // 图像数据大小（字节）
    int32_t width;            // 图像宽度
    int32_t height;           // 图像高度
    QuadPoints quad;          // 检测到的四边形顶点
    int32_t success;          // 1=成功, 0=失败
    char error_message[256];  // 错误信息
} ScanResult;

// 算法参数结构
typedef struct {
    double canny_threshold1;      // Canny 边缘检测阈值1 (默认 50)
    double canny_threshold2;      // Canny 边缘检测阈值2 (默认 150)
    int32_t gaussian_kernel_size; // 高斯模糊核大小 (默认 5)
    double min_contour_area_ratio;// 最小轮廓面积比例 (默认 0.1)
    int32_t jpeg_quality;         // JPEG 质量 (默认 95)
} ScannerParams;

/**
 * 获取默认参数
 */
SCANNER_API ScannerParams scanner_get_default_params(void);

/**
 * 处理文档扫描
 *
 * @param input_data    输入图像数据（JPEG/PNG 编码）
 * @param input_size    输入数据大小
 * @param apply_enhancement  是否应用图像增强 (1=是, 0=否)
 * @param params        算法参数
 * @param result        输出结果（调用者需调用 scanner_free_result 释放）
 * @return              0=成功, 非0=失败
 */
SCANNER_API int32_t scanner_process_scan(
    const uint8_t* input_data,
    int32_t input_size,
    int32_t apply_enhancement,
    const ScannerParams* params,
    ScanResult* result
);

/**
 * 仅检测文档边界（不做透视变换）
 *
 * @param input_data    输入图像数据
 * @param input_size    输入数据大小
 * @param params        算法参数
 * @param quad          输出四边形顶点
 * @return              1=检测到, 0=未检测到
 */
SCANNER_API int32_t scanner_detect_bounds(
    const uint8_t* input_data,
    int32_t input_size,
    const ScannerParams* params,
    QuadPoints* quad
);

/**
 * 应用透视变换（使用已知的四边形顶点）
 *
 * @param input_data    输入图像数据
 * @param input_size    输入数据大小
 * @param quad          四边形顶点
 * @param apply_enhancement  是否应用图像增强
 * @param jpeg_quality  JPEG 质量
 * @param result        输出结果
 * @return              0=成功, 非0=失败
 */
SCANNER_API int32_t scanner_apply_transform(
    const uint8_t* input_data,
    int32_t input_size,
    const QuadPoints* quad,
    int32_t apply_enhancement,
    int32_t jpeg_quality,
    ScanResult* result
);

/**
 * 释放扫描结果中的图像数据
 */
SCANNER_API void scanner_free_result(ScanResult* result);

/**
 * 获取库版本信息
 */
SCANNER_API const char* scanner_get_version(void);

#ifdef __cplusplus
}
#endif

#endif // OPENCV_SCANNER_H
