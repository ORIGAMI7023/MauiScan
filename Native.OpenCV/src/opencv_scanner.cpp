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
    params.canny_threshold1 = 30.0;   // 降低阈值，更容易检测边缘
    params.canny_threshold2 = 100.0;  // 降低阈值
    params.gaussian_kernel_size = 5;
    params.min_contour_area_ratio = 0.05; // 降低最小面积要求
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

// 候选轮廓结构体
struct ContourCandidate {
    std::vector<Point> points;
    double score;
    double area;
};

// 计算轮廓置信度评分
static double calculate_contour_score(
    const std::vector<Point>& contour,
    const Mat& /* edges */,  // 保留参数以便未来扩展
    double image_area
) {
    double score = 0.0;

    // 1. 面积分数 (0-30分)：面积占比越大越好（降低权重）
    double area = contourArea(contour);
    double area_ratio = area / image_area;
    score += std::min(area_ratio * 100.0, 30.0);

    // 2. 四边形拟合度 (0-40分)：越接近四边形越好（提高权重，支持透视变形）
    double peri = arcLength(contour, true);
    std::vector<Point> approx;
    approxPolyDP(contour, approx, 0.03 * peri, true);

    if (approx.size() == 4) {
        score += 40.0; // 完美四边形（包括透视拍摄的梯形）
    } else if (approx.size() == 5 || approx.size() == 6) {
        score += 25.0; // 接近四边形
    } else if (approx.size() >= 7 && approx.size() <= 10) {
        score += 12.0; // 勉强可用
    }

    // 3. 凸性分数 (0-15分)：凸多边形更可能是文档
    if (isContourConvex(approx)) {
        score += 15.0;
    }

    // 4. 边缘清晰度 (0-15分)：轮廓周长与面积比
    double compactness = (peri * peri) / area;
    double normalized_compactness = std::min(compactness / 50.0, 1.0);
    score += (1.0 - normalized_compactness) * 15.0;

    return score;
}

// 检测文档边界（智能评分算法）
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

        // 3. 轻微增强对比度（提高边缘检测效果）
        Mat enhanced;
        blurred.convertTo(enhanced, -1, 1.15, 0); // alpha=1.15 (对比度), beta=0 (亮度)

        // 4. Canny 边缘检测
        Mat edges;
        Canny(enhanced, edges, params->canny_threshold1, params->canny_threshold2);

        // 5. 查找轮廓
        std::vector<std::vector<Point>> contours;
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            return false;
        }

        // 6. 使用低阈值筛选候选轮廓（3%，避免遗漏）
        double image_area = image.cols * image.rows;
        double min_area = image_area * 0.03; // 降低到 3%

        std::vector<ContourCandidate> candidates;

        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area < min_area) {
                continue;
            }

            // 计算置信度评分
            double score = calculate_contour_score(contour, edges, image_area);

            ContourCandidate candidate;
            candidate.points = contour;
            candidate.score = score;
            candidate.area = area;

            candidates.push_back(candidate);
        }

        if (candidates.empty()) {
            return false;
        }

        // 7. 按评分降序排序
        std::sort(candidates.begin(), candidates.end(),
            [](const ContourCandidate& a, const ContourCandidate& b) {
                return a.score > b.score;
            });

        // 8. 选择最高分的候选轮廓
        const auto& best_candidate = candidates[0];

        // 9. 置信度阈值检查：至少 38 分才认为是有效文档（降低阈值提高识别率）
        if (best_candidate.score < 38.0) {
            return false;
        }

        // 10. 尝试将最佳候选轮廓近似为四边形
        double peri = arcLength(best_candidate.points, true);
        std::vector<Point> approx;
        approxPolyDP(best_candidate.points, approx, 0.03 * peri, true);

        if (approx.size() == 4) {
            sort_quad_points(approx, quad);
            return true;
        }

        // 11. 如果不是四边形，尝试更宽松的近似
        if (approx.size() >= 4 && approx.size() <= 6) {
            std::vector<Point> approx2;
            approxPolyDP(best_candidate.points, approx2, 0.05 * peri, true);
            if (approx2.size() == 4) {
                sort_quad_points(approx2, quad);
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

    // 初始化 result 结构体
    result->image_data = nullptr;
    result->image_size = 0;
    result->width = 0;
    result->height = 0;
    result->success = 0;
    std::memset(&result->quad, 0, sizeof(QuadPoints));
    result->error_message[0] = '\0';

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
        bool bounds_detected = detect_document_bounds_internal(original, params, &quad);

        Mat final_image;

        if (bounds_detected) {
            // 2. 透视变换
            Mat warped;
            if (!apply_perspective_transform_internal(original, &quad, warped)) {
                // 透视变换失败，使用原图
                final_image = original;
            } else {
                // 3. 可选图像增强
                if (apply_enhancement) {
                    apply_enhancement_internal(warped, final_image);
                } else {
                    final_image = warped;
                }
            }
        } else {
            // 未检测到文档边界，返回原图
            final_image = original;
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

// ========== Corner Refinement Implementation ==========

// 辅助函数：精修单个角点
static bool refine_single_corner(
    const Mat& gray_image,
    float ml_x,
    float ml_y,
    float& refined_x,
    float& refined_y
) {
    const int patch_size = 64; // 搜索窗口大小
    const int half_patch = patch_size / 2;

    try {
        // 1. 裁剪 patch（确保不越界）
        int center_x = static_cast<int>(ml_x);
        int center_y = static_cast<int>(ml_y);

        int x1 = std::max(0, center_x - half_patch);
        int y1 = std::max(0, center_y - half_patch);
        int x2 = std::min(gray_image.cols, center_x + half_patch);
        int y2 = std::min(gray_image.rows, center_y + half_patch);

        if (x2 - x1 < 20 || y2 - y1 < 20) {
            return false; // Patch 太小
        }

        Mat patch = gray_image(Rect(x1, y1, x2 - x1, y2 - y1));

        // 2. Canny 边缘检测
        Mat edges;
        Canny(patch, edges, 50, 150, 3);

        // 3. 霍夫直线检测
        std::vector<Vec4i> lines;
        HoughLinesP(edges, lines, 1, CV_PI / 180, 30, 20, 5);

        if (lines.size() < 2) {
            return false; // 检测到的直线太少
        }

        // 4. 直线聚类（水平 vs 垂直）
        std::vector<Vec4i> horizontal_lines;
        std::vector<Vec4i> vertical_lines;

        for (const auto& line : lines) {
            int dx = std::abs(line[2] - line[0]);
            int dy = std::abs(line[3] - line[1]);

            if (dx > dy) {
                horizontal_lines.push_back(line);
            } else {
                vertical_lines.push_back(line);
            }
        }

        if (horizontal_lines.empty() || vertical_lines.empty()) {
            return false; // 没有找到两组直线
        }

        // 5. 拟合直线（最小二乘法）
        auto fit_line = [](const std::vector<Vec4i>& lines) -> std::pair<float, float> {
            std::vector<Point2f> points;
            for (const auto& line : lines) {
                points.push_back(Point2f(line[0], line[1]));
                points.push_back(Point2f(line[2], line[3]));
            }

            float avg_x = 0, avg_y = 0;
            for (const auto& p : points) {
                avg_x += p.x;
                avg_y += p.y;
            }
            avg_x /= points.size();
            avg_y /= points.size();

            float numerator = 0, denominator = 0;
            for (const auto& p : points) {
                numerator += (p.x - avg_x) * (p.y - avg_y);
                denominator += (p.x - avg_x) * (p.x - avg_x);
            }

            if (std::abs(denominator) < 1e-6) {
                return {0, 0}; // 失败
            }

            float k = numerator / denominator;
            float b = avg_y - k * avg_x;
            return {k, b};
        };

        auto h_line = fit_line(horizontal_lines);
        auto v_line = fit_line(vertical_lines);

        if (std::abs(h_line.first) < 1e-6 || std::abs(v_line.first) < 1e-6) {
            return false;
        }

        // 6. 计算交点
        // h_line: y = k1*x + b1
        // v_line: y = k2*x + b2
        float k1 = h_line.first, b1 = h_line.second;
        float k2 = v_line.first, b2 = v_line.second;

        if (std::abs(k1 - k2) < 1e-6) {
            return false; // 平行线
        }

        float x = (b2 - b1) / (k1 - k2);
        float y = k1 * x + b1;

        // 7. 转换回原图坐标
        refined_x = x1 + x;
        refined_y = y1 + y;

        // 8. 验证精修结果（距离 ML 预测不能太远）
        float distance = std::sqrt(
            (refined_x - ml_x) * (refined_x - ml_x) +
            (refined_y - ml_y) * (refined_y - ml_y)
        );

        if (distance > patch_size) {
            return false; // 超出搜索范围
        }

        return true;
    } catch (const cv::Exception&) {
        return false;
    }
}

int32_t scanner_refine_corners(
    const uint8_t* input_data,
    int32_t input_size,
    const QuadPointsF* ml_quad,
    QuadPointsF* refined_quad
) {
    if (!input_data || input_size <= 0 || !ml_quad || !refined_quad) {
        return 0;
    }

    try {
        // 解码图像为灰度图
        std::vector<uint8_t> buffer(input_data, input_data + input_size);
        Mat image = imdecode(buffer, IMREAD_GRAYSCALE);

        if (image.empty()) {
            // 解码失败，直接返回原始坐标
            *refined_quad = *ml_quad;
            return 0;
        }

        // 初始化为 ML 预测的坐标
        *refined_quad = *ml_quad;

        int success_count = 0;

        // 精修左上角
        float tl_x, tl_y;
        if (refine_single_corner(image, ml_quad->top_left_x, ml_quad->top_left_y, tl_x, tl_y)) {
            refined_quad->top_left_x = tl_x;
            refined_quad->top_left_y = tl_y;
            success_count++;
        }

        // 精修右上角
        float tr_x, tr_y;
        if (refine_single_corner(image, ml_quad->top_right_x, ml_quad->top_right_y, tr_x, tr_y)) {
            refined_quad->top_right_x = tr_x;
            refined_quad->top_right_y = tr_y;
            success_count++;
        }

        // 精修右下角
        float br_x, br_y;
        if (refine_single_corner(image, ml_quad->bottom_right_x, ml_quad->bottom_right_y, br_x, br_y)) {
            refined_quad->bottom_right_x = br_x;
            refined_quad->bottom_right_y = br_y;
            success_count++;
        }

        // 精修左下角
        float bl_x, bl_y;
        if (refine_single_corner(image, ml_quad->bottom_left_x, ml_quad->bottom_left_y, bl_x, bl_y)) {
            refined_quad->bottom_left_x = bl_x;
            refined_quad->bottom_left_y = bl_y;
            success_count++;
        }

        // 如果至少成功精修 2 个角点，认为精修成功
        return (success_count >= 2) ? 1 : 0;
    } catch (const cv::Exception&) {
        // 精修失败，返回原始坐标
        *refined_quad = *ml_quad;
        return 0;
    } catch (...) {
        *refined_quad = *ml_quad;
        return 0;
    }
}
