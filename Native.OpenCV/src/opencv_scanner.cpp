/**
 * opencv_scanner.cpp - OpenCV Document Scanner Implementation
 *
 * 核心算法：灰度化 → 高斯模糊 → Canny边缘检测 → 轮廓查找 → 四边形筛选 → 透视变换
 */

#include "opencv_scanner.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>

using namespace cv;

// 版本信息
static const char* SCANNER_VERSION = "1.0.0";

// 默认参数
ScannerParams scanner_get_default_params(void) {
    ScannerParams params;
    params.canny_threshold1 = 50.0;
    params.canny_threshold2 = 150.0;
    params.gaussian_kernel_size = 5;
    params.min_contour_area_ratio = 0.1;
    params.jpeg_quality = 95;
    return params;
}

// 计算两点间距离
static double calculate_distance(const Point2f& p1, const Point2f& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// 排序四边形顶点（确保顺序：左上、右上、右下、左下）
static void sort_quad_points(const std::vector<Point>& points, QuadPoints* quad) {
    if (points.size() != 4) return;

    // 按 y 坐标排序，分为上下两组
    std::vector<Point> sorted = points;
    std::sort(sorted.begin(), sorted.end(), [](const Point& a, const Point& b) {
        return a.y < b.y;
    });

    std::vector<Point> top_points(sorted.begin(), sorted.begin() + 2);
    std::vector<Point> bottom_points(sorted.begin() + 2, sorted.end());

    // 按 x 坐标排序
    std::sort(top_points.begin(), top_points.end(), [](const Point& a, const Point& b) {
        return a.x < b.x;
    });
    std::sort(bottom_points.begin(), bottom_points.end(), [](const Point& a, const Point& b) {
        return a.x < b.x;
    });

    // 填充结果
    quad->top_left_x = top_points[0].x;
    quad->top_left_y = top_points[0].y;
    quad->top_right_x = top_points[1].x;
    quad->top_right_y = top_points[1].y;
    quad->bottom_right_x = bottom_points[1].x;
    quad->bottom_right_y = bottom_points[1].y;
    quad->bottom_left_x = bottom_points[0].x;
    quad->bottom_left_y = bottom_points[0].y;
}

// 检测文档边界
static bool detect_document_bounds_internal(
    const Mat& image,
    const ScannerParams* params,
    QuadPoints* quad
) {
    try {
        // 1. 转灰度
        Mat gray;
        if (image.channels() == 3) {
            cvtColor(image, gray, COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // 2. 高斯模糊（降噪）
        Mat blurred;
        int kernel_size = params->gaussian_kernel_size;
        if (kernel_size % 2 == 0) kernel_size++; // 确保为奇数
        GaussianBlur(gray, blurred, Size(kernel_size, kernel_size), 0);

        // 3. Canny 边缘检测
        Mat edges;
        Canny(blurred, edges, params->canny_threshold1, params->canny_threshold2);

        // 4. 查找轮廓
        std::vector<std::vector<Point>> contours;
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            return false;
        }

        // 5. 筛选最大四边形轮廓
        double image_area = image.cols * image.rows;
        double min_area = image_area * params->min_contour_area_ratio;

        // 按面积降序排序
        std::sort(contours.begin(), contours.end(), [](const std::vector<Point>& a, const std::vector<Point>& b) {
            return contourArea(a) > contourArea(b);
        });

        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area < min_area) {
                break;
            }

            // 轮廓近似
            double peri = arcLength(contour, true);
            std::vector<Point> approx;
            approxPolyDP(contour, approx, 0.02 * peri, true);

            // 如果是四边形，则认为是文档边界
            if (approx.size() == 4) {
                sort_quad_points(approx, quad);
                return true;
            }
        }

        return false;
    } catch (const cv::Exception& e) {
        return false;
    }
}

// 应用透视变换
static bool apply_perspective_transform_internal(
    const Mat& source,
    const QuadPoints* quad,
    Mat& output
) {
    try {
        // 源四边形顶点
        std::vector<Point2f> src_points = {
            Point2f(quad->top_left_x, quad->top_left_y),
            Point2f(quad->top_right_x, quad->top_right_y),
            Point2f(quad->bottom_right_x, quad->bottom_right_y),
            Point2f(quad->bottom_left_x, quad->bottom_left_y)
        };

        // 计算目标矩形尺寸
        double width_top = calculate_distance(src_points[0], src_points[1]);
        double width_bottom = calculate_distance(src_points[3], src_points[2]);
        double height_left = calculate_distance(src_points[0], src_points[3]);
        double height_right = calculate_distance(src_points[1], src_points[2]);

        int max_width = static_cast<int>(std::max(width_top, width_bottom));
        int max_height = static_cast<int>(std::max(height_left, height_right));

        // 目标矩形顶点
        std::vector<Point2f> dst_points = {
            Point2f(0, 0),
            Point2f(max_width - 1, 0),
            Point2f(max_width - 1, max_height - 1),
            Point2f(0, max_height - 1)
        };

        // 获取透视变换矩阵
        Mat transform_matrix = getPerspectiveTransform(src_points, dst_points);

        // 应用变换
        warpPerspective(source, output, transform_matrix, Size(max_width, max_height));

        return true;
    } catch (const cv::Exception& e) {
        return false;
    }
}

// 应用图像增强
static void apply_enhancement_internal(const Mat& input, Mat& output) {
    try {
        Mat gray;
        if (input.channels() == 3) {
            cvtColor(input, gray, COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }

        // 自适应阈值（增强对比度）
        adaptiveThreshold(gray, output, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    } catch (const cv::Exception& e) {
        output = input.clone();
    }
}

// 公开 API 实现

int32_t scanner_process_scan(
    const uint8_t* input_data,
    int32_t input_size,
    int32_t apply_enhancement,
    const ScannerParams* params,
    ScanResult* result
) {
    if (!input_data || input_size <= 0 || !params || !result) {
        return -1;
    }

    try {
        // 解码输入图像
        std::vector<uint8_t> buffer(input_data, input_data + input_size);
        Mat original = imdecode(buffer, IMREAD_COLOR);

        if (original.empty()) {
            std::strcpy(result->error_message, "Failed to decode image");
            result->success = 0;
            return -1;
        }

        // 1. 检测文档边界
        QuadPoints quad;
        if (!detect_document_bounds_internal(original, params, &quad)) {
            std::strcpy(result->error_message, "No document bounds detected");
            result->success = 0;
            return -1;
        }

        // 2. 透视变换
        Mat warped;
        if (!apply_perspective_transform_internal(original, &quad, warped)) {
            std::strcpy(result->error_message, "Perspective transform failed");
            result->success = 0;
            return -1;
        }

        // 3. 可选图像增强
        Mat final_image;
        if (apply_enhancement) {
            apply_enhancement_internal(warped, final_image);
        } else {
            final_image = warped;
        }

        // 4. 编码为 JPEG
        std::vector<uint8_t> output_buffer;
        std::vector<int> encode_params = {IMWRITE_JPEG_QUALITY, params->jpeg_quality};
        if (!imencode(".jpg", final_image, output_buffer, encode_params)) {
            std::strcpy(result->error_message, "JPEG encoding failed");
            result->success = 0;
            return -1;
        }

        // 5. 分配输出内存
        result->image_size = static_cast<int32_t>(output_buffer.size());
        result->image_data = new uint8_t[result->image_size];
        std::memcpy(result->image_data, output_buffer.data(), result->image_size);
        result->width = final_image.cols;
        result->height = final_image.rows;
        result->quad = quad;
        result->success = 1;
        result->error_message[0] = '\0';

        return 0;
    } catch (const cv::Exception& e) {
        std::snprintf(result->error_message, sizeof(result->error_message), "OpenCV error: %s", e.what());
        result->success = 0;
        return -1;
    } catch (const std::exception& e) {
        std::snprintf(result->error_message, sizeof(result->error_message), "Exception: %s", e.what());
        result->success = 0;
        return -1;
    }
}

int32_t scanner_detect_bounds(
    const uint8_t* input_data,
    int32_t input_size,
    const ScannerParams* params,
    QuadPoints* quad
) {
    if (!input_data || input_size <= 0 || !params || !quad) {
        return 0;
    }

    try {
        std::vector<uint8_t> buffer(input_data, input_data + input_size);
        Mat image = imdecode(buffer, IMREAD_COLOR);

        if (image.empty()) {
            return 0;
        }

        return detect_document_bounds_internal(image, params, quad) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int32_t scanner_apply_transform(
    const uint8_t* input_data,
    int32_t input_size,
    const QuadPoints* quad,
    int32_t apply_enhancement,
    int32_t jpeg_quality,
    ScanResult* result
) {
    if (!input_data || input_size <= 0 || !quad || !result) {
        return -1;
    }

    try {
        std::vector<uint8_t> buffer(input_data, input_data + input_size);
        Mat original = imdecode(buffer, IMREAD_COLOR);

        if (original.empty()) {
            std::strcpy(result->error_message, "Failed to decode image");
            result->success = 0;
            return -1;
        }

        // 透视变换
        Mat warped;
        if (!apply_perspective_transform_internal(original, quad, warped)) {
            std::strcpy(result->error_message, "Perspective transform failed");
            result->success = 0;
            return -1;
        }

        // 可选增强
        Mat final_image;
        if (apply_enhancement) {
            apply_enhancement_internal(warped, final_image);
        } else {
            final_image = warped;
        }

        // 编码
        std::vector<uint8_t> output_buffer;
        std::vector<int> encode_params = {IMWRITE_JPEG_QUALITY, jpeg_quality};
        if (!imencode(".jpg", final_image, output_buffer, encode_params)) {
            std::strcpy(result->error_message, "JPEG encoding failed");
            result->success = 0;
            return -1;
        }

        result->image_size = static_cast<int32_t>(output_buffer.size());
        result->image_data = new uint8_t[result->image_size];
        std::memcpy(result->image_data, output_buffer.data(), result->image_size);
        result->width = final_image.cols;
        result->height = final_image.rows;
        result->quad = *quad;
        result->success = 1;
        result->error_message[0] = '\0';

        return 0;
    } catch (const cv::Exception& e) {
        std::snprintf(result->error_message, sizeof(result->error_message), "OpenCV error: %s", e.what());
        result->success = 0;
        return -1;
    } catch (...) {
        std::strcpy(result->error_message, "Unknown error");
        result->success = 0;
        return -1;
    }
}

void scanner_free_result(ScanResult* result) {
    if (result && result->image_data) {
        delete[] result->image_data;
        result->image_data = nullptr;
        result->image_size = 0;
    }
}

const char* scanner_get_version(void) {
    return SCANNER_VERSION;
}
