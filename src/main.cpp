#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <glob.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <omp.h>
#include "hailo/hailort.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "simpletrack/simpletrack.hpp"
#include "simpletrack/boundary.hpp"
#include "../include/hailo_inference.h"
#include "../include/types.h"

using namespace hailort;

namespace {

bool g_debug_coords = false;
bool g_debug_det = false;
bool g_debug_yaw = false;

struct Args {
    std::string data_glob = "at128/*.bin";
    std::string config_path = "SimpleTrack/configs/waymo_configs/vc_kf_giou.yaml";
    std::string class_config_path = "SimpleTrack/configs/tracker_params.yaml";
    std::string video_path = "result.mp4";
    double video_fps = 10.0;
    bool play = true;
    bool no_gui = true;
    std::optional<int> max_frames;
    int start_frame = 0;
    std::optional<int> end_frame;
    bool debug_assoc = false;
    bool debug_life = false;
    bool debug_all = false;
    bool vis_detections = true;
    bool vis_tracks = true;
    bool debug_coords = false;
    bool debug_det = false;
    bool debug_with_hailo = false;
    bool debug_yaw = false;
};

static bool parse_bool(const std::string& s) {
    std::string t;
    t.reserve(s.size());
    for (char c : s) t.push_back(std::tolower(static_cast<unsigned char>(c)));
    if (t == "1" || t == "true" || t == "yes" || t == "on") return true;
    if (t == "0" || t == "false" || t == "no" || t == "off") return false;
    throw std::runtime_error("Invalid bool: " + s + " (use true/false)");
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for option: " + option);
            }
            return argv[++i];
        };

        if (key == "--data-glob") {
            args.data_glob = need_value(key);
        } else if (key == "--config-path") {
            args.config_path = need_value(key);
        } else if (key == "--class-config") {
            args.class_config_path = need_value(key);
        } else if (key == "--video-path") {
            args.video_path = need_value(key);
        } else if (key == "--video-fps") {
            args.video_fps = std::stod(need_value(key));
        } else if (key == "--play") {
            args.play = true;
        } else if (key == "--no-gui") {
            args.no_gui = true;
        } else if (key == "--max-frames") {
            args.max_frames = std::stoi(need_value(key));
        } else if (key == "--start-frame") {
            args.start_frame = std::stoi(need_value(key));
        } else if (key == "--end-frame") {
            args.end_frame = std::stoi(need_value(key));
        } else if (key == "--debug-assoc") {
            args.debug_assoc = true;
        } else if (key == "--debug-life") {
            args.debug_life = true;
        } else if (key == "--debug-all") {
            args.debug_all = true;
        } else if (key == "--detection-vis") {
            args.vis_detections = parse_bool(need_value(key));
        } else if (key == "--tracking-vis") {
            args.vis_tracks = parse_bool(need_value(key));
        } else if (key == "--debug-coords") {
            args.debug_coords = true;
        } else if (key == "--debug-det") {
            args.debug_det = true;
        } else if (key == "--debug-with-hailo") {
            args.debug_with_hailo = true;
        } else if (key == "--debug-yaw") {
            args.debug_yaw = true;
        } else if (key == "--help" || key == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --data-glob <pattern|dir|file>  LiDAR .bin glob pattern or path\n"
                      << "  --config-path <path>            Tracker config YAML path\n"
                      << "  --class-config <path>           Class-specific tracker params YAML\n"
                      << "  --video-path <path>             Output video path (mp4)\n"
                      << "  --video-fps <fps>               Output video FPS (default 10)\n"
                      << "  --play                          Disable pause between frames\n"
                      << "  --no-gui                        Run without OpenCV window\n"
                      << "  --max-frames <N>                Limit processed frames\n"
                      << "  --start-frame <idx>             Start from this sorted frame index (0-based)\n"
                      << "  --end-frame <idx>               End at this sorted frame index (inclusive, 0-based)\n"
                      << "  --debug-assoc                   Enable association debug logs\n"
                      << "  --debug-life                    Enable lifecycle debug logs\n"
                      << "  --debug-all                     Enable both debug logs\n"
                      << "  --detection-vis <bool>          Draw raw detections (default true)\n"
                      << "  --tracking-vis <bool>           Draw tracking boxes (default true)\n"
                      << "  --debug-coords                  Log BEV->ego coordinate conversions\n"
                      << "  --debug-det                     Log detection yaw conversions\n"
                      << "  --debug-with-hailo              Extra logs to align with Hailo pipeline\n"
                      << "  --debug-yaw                     Log raw decoded boxes (pre-NMS) with yaw\n";
            return false;
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }
    return true;
}

// Constants lifted from SimpleTrack/tools/inference_sfa3d_at128_onnx.cpp
constexpr int GRID = 1216;
constexpr int HEATMAP_H = 304;
constexpr int HEATMAP_W = 304;
constexpr int NUM_CLASSES = 3;
constexpr int DOWN_RATIO = 4;
constexpr float CENTER_PEAK_THRESH = 0.4f;
constexpr float MAX_Z_M = 5.5f;
constexpr float DIM_SCALE = 10.0f;
constexpr int TOPK = 60;
constexpr float DT = 0.1f;

const std::array<cv::Scalar, 8> COLORS = {
    cv::Scalar(255, 255, 0),  cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),   cv::Scalar(0, 120, 255),
    cv::Scalar(120, 120, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 255, 120), cv::Scalar(255, 0, 120)
};

struct Point {
    float x;
    float y;
    float z;
    float intensity;
};

struct DecodeResult {
    float score;
    float x;  // center (feature map)
    float y;
    float z;
    float h;
    float w;
    float l;
    float dirx;
    float diry;
    float cls;
};

struct ProcessedDetection {
    float score;
    float x_px;
    float y_px;
    float z_m;
    float h_m;
    float w_px;
    float l_px;
    float yaw;
    int cls;
};

struct TimingStats {
    double sum{0.0};
    double min{std::numeric_limits<double>::max()};
    double max{0.0};
};

inline void update_stats(TimingStats &stats, double val) {
    stats.sum += val;
    if (val < stats.min) stats.min = val;
    if (val > stats.max) stats.max = val;
}

struct CellInfo {
    float max_z = -std::numeric_limits<float>::infinity();
    float intensity = 0.0f;
    int count = 0;
    bool has_value = false;
};

std::vector<std::string> list_bin_files(const std::filesystem::path& dir) {
    std::vector<std::string> files;
    if (!std::filesystem::exists(dir)) {
        return files;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".bin") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<std::string> glob_files(const std::string& pattern) {
    glob_t glob_result;
    std::vector<std::string> files;
    int ret = glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);
    if (ret != 0) {
        globfree(&glob_result);
        return files;
    }
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        files.emplace_back(glob_result.gl_pathv[i]);
    }
    globfree(&glob_result);
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<std::string> resolve_bin_inputs(const std::string& pattern) {
    std::filesystem::path p(pattern);
    if (std::filesystem::is_regular_file(p)) {
        return {p.string()};
    }
    if (std::filesystem::is_directory(p)) {
        return list_bin_files(p);
    }
    return glob_files(pattern);
}

std::vector<Point> load_point_cloud(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open LiDAR file: " + path);
    }
    ifs.seekg(0, std::ios::end);
    std::streamsize bytes = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    if (bytes % (sizeof(float) * 4) != 0) {
        throw std::runtime_error("Invalid point cloud file (size mismatch): " + path);
    }
    size_t count = static_cast<size_t>(bytes) / (sizeof(float) * 4);
    std::vector<Point> cloud(count);
    for (size_t i = 0; i < count; ++i) {
        float data[4];
        ifs.read(reinterpret_cast<char*>(data), sizeof(data));
        cloud[i] = Point{data[0], data[1], data[2], data[3]};
    }
    return cloud;
}

std::vector<Point> remove_points(const std::vector<Point>& cloud, const simpletrack::Boundary& boundary) {
    std::vector<Point> out;
    out.reserve(cloud.size());
    for (const auto& p : cloud) {
        if (p.x < boundary.minX || p.x > boundary.maxX) continue;
        if (p.y < boundary.minY || p.y > boundary.maxY) continue;
        if (p.z < boundary.minZ || p.z > boundary.maxZ) continue;
        Point q = p;
        q.z -= boundary.minZ;
        out.emplace_back(q);
    }
    return out;
}

std::vector<uint8_t> make_bev_map_int8(const std::vector<Point>& cloud, const simpletrack::Boundary& boundary) {
    const int H = GRID + 1;
    const int W = GRID + 1;
    const double d = boundary.discretization();
    const double inv_d = 1.0 / d;
    const double z_range = boundary.z_range();
    const int tile_size = 32;
    const int tile_rows = (H + tile_size - 1) / tile_size;
    const int tile_cols = (W + tile_size - 1) / tile_size;
    const int num_tiles = tile_rows * tile_cols;

    struct TilePoint {
        int xi;
        int yi;
        float z;
        float intensity;
    };

    static std::vector<std::vector<TilePoint>> tile_points;
    if (tile_points.size() != static_cast<size_t>(num_tiles)) {
        tile_points.clear();
        tile_points.resize(num_tiles);
    }
    for (auto &vec : tile_points) vec.clear();

    for (const auto& p : cloud) {
        if (p.z < 0.f || p.z > MAX_Z_M) continue;
        int xi = static_cast<int>(p.x * inv_d + H / 4.0);
        int yi = static_cast<int>(p.y * inv_d + W / 2.0);
        if (xi < 0 || xi >= H || yi < 0 || yi >= W) continue;
        int tile_r = xi / tile_size;
        int tile_c = yi / tile_size;
        int tile_id = tile_r * tile_cols + tile_c;
        tile_points[tile_id].push_back(TilePoint{xi, yi, p.z, p.intensity});
    }

    static std::vector<float> max_z;
    static std::vector<float> intensity;
    static std::vector<int> counts;
    static std::vector<uint32_t> cell_frame;
    static uint32_t frame_token = 1;
    const size_t total_cells = static_cast<size_t>(H) * static_cast<size_t>(W);
    if (max_z.size() != total_cells) {
        max_z.assign(total_cells, 0.0f);
        intensity.assign(total_cells, 0.0f);
        counts.assign(total_cells, 0);
        cell_frame.assign(total_cells, 0);
    }
    frame_token += 1;
    if (frame_token == 0) {
        frame_token = 1;
        std::fill(cell_frame.begin(), cell_frame.end(), 0);
    }

    int max_threads = omp_get_max_threads();
    std::vector<std::vector<size_t>> thread_used(static_cast<size_t>(max_threads));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto &local_used = thread_used[tid];
        local_used.clear();
#pragma omp for schedule(dynamic)
        for (int tile = 0; tile < num_tiles; ++tile) {
            const auto& pts = tile_points[tile];
            if (pts.empty()) continue;
            for (const auto& tp : pts) {
                size_t idx = static_cast<size_t>(tp.xi) * static_cast<size_t>(W) + static_cast<size_t>(tp.yi);
                if (cell_frame[idx] != frame_token) {
                    cell_frame[idx] = frame_token;
                    max_z[idx] = tp.z;
                    intensity[idx] = tp.intensity;
                    counts[idx] = 1;
                    local_used.push_back(idx);
                } else {
                    if (tp.z > max_z[idx]) {
                        max_z[idx] = tp.z;
                        intensity[idx] = tp.intensity;
                    }
                    counts[idx] += 1;
                }
            }
        }
    }

    static std::vector<size_t> used_indices;
    used_indices.clear();
    size_t total_used = 0;
    for (const auto& vec : thread_used) total_used += vec.size();
    used_indices.reserve(total_used);
    for (auto& vec : thread_used) {
        used_indices.insert(used_indices.end(), vec.begin(), vec.end());
    }

    static std::vector<uint8_t> bev;
    bev.assign(static_cast<size_t>(GRID * GRID * 3), 0);
    static std::vector<uint8_t> dens_lut;
    if (dens_lut.empty()) {
        constexpr int kMaxCount = 4096;
        dens_lut.resize(kMaxCount + 1);
        for (int c = 0; c <= kMaxCount; ++c) {
            float dens = std::min(1.0f, std::log(static_cast<float>(c) + 1.0f) / std::log(128.0f));
            dens_lut[c] = static_cast<uint8_t>(dens * 255.0f);
        }
    }

    for (size_t idx : used_indices) {
        size_t xi = idx / static_cast<size_t>(W);
        size_t yi = idx % static_cast<size_t>(W);
        if (xi >= static_cast<size_t>(GRID) || yi >= static_cast<size_t>(GRID)) continue;

        float normalized_z = max_z[idx] / static_cast<float>(z_range);
        if (normalized_z < 0.0f) normalized_z = 0.0f;
        else if (normalized_z > 1.0f) normalized_z = 1.0f;
        float inten = intensity[idx];
        if (inten < 0.0f) inten = 0.0f;
        else if (inten > 255.0f) inten = 255.0f;
        int dens_idx = counts[idx];
        if (dens_idx < 0) dens_idx = 0;
        if (dens_idx >= static_cast<int>(dens_lut.size())) dens_idx = static_cast<int>(dens_lut.size()) - 1;

        size_t dst_idx = (xi * static_cast<size_t>(GRID) + yi) * 3;
        bev[dst_idx + 0] = static_cast<uint8_t>(inten);
        bev[dst_idx + 1] = static_cast<uint8_t>(normalized_z * 255.0f);
        bev[dst_idx + 2] = dens_lut[dens_idx];
    }
    return bev;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<float> nms_heatmap(const std::vector<float>& heatmap, int classes, int H, int W) {
    std::vector<float> out(heatmap.size(), 0.0f);
    auto idx = [&](int c, int y, int x) {
        return ((c * H) + y) * W + x;
    };
    for (int c = 0; c < classes; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float val = heatmap[idx(c, y, x)];
                bool keep = true;
                for (int dy = -1; dy <= 1 && keep; ++dy) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= H) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = x + dx;
                        if (nx < 0 || nx >= W) continue;
                        if (heatmap[idx(c, ny, nx)] > val) {
                            keep = false;
                            break;
                        }
                    }
                }
                if (keep) {
                    out[idx(c, y, x)] = val;
                }
            }
        }
    }
    return out;
}

std::array<cv::Point, 4> get_corners(float x, float y, float w, float l, float yaw) {
    float cos_y = std::cos(yaw);
    float sin_y = std::sin(yaw);
    return {
        cv::Point(cvRound(x - w / 2 * cos_y - l / 2 * sin_y), cvRound(y - w / 2 * sin_y + l / 2 * cos_y)),
        cv::Point(cvRound(x - w / 2 * cos_y + l / 2 * sin_y), cvRound(y - w / 2 * sin_y - l / 2 * cos_y)),
        cv::Point(cvRound(x + w / 2 * cos_y + l / 2 * sin_y), cvRound(y + w / 2 * sin_y - l / 2 * cos_y)),
        cv::Point(cvRound(x + w / 2 * cos_y - l / 2 * sin_y), cvRound(y + w / 2 * sin_y + l / 2 * cos_y))
    };
}

void draw_rotated_box(cv::Mat& img, float x, float y, float w, float l, float yaw, const cv::Scalar& color, int thickness = 2) {
    auto corners = get_corners(x, y, w, l, yaw);
    std::vector<cv::Point> pts(corners.begin(), corners.end());
    cv::polylines(img, pts, true, color, thickness);
    cv::line(img, corners[0], corners[3], cv::Scalar(255, 255, 0), thickness);
}

std::vector<DecodeResult> decode(const std::vector<float>& hm,
                                 const std::vector<float>& offset,
                                 const std::vector<float>& direction,
                                 const std::vector<float>& zcoor,
                                 const std::vector<float>& dim,
                                 int classes,
                                 int H,
                                 int W,
                                 int K) {
    auto idx = [&](const std::vector<float>& vec, int c, int y, int x, int channels = 1, int ch = 0) {
        if (channels == 1) {
            return vec[((c * H) + y) * W + x];
        }
        return vec[(((ch * H) + y) * W + x)];
    };

    std::vector<float> hm_sig(hm.size());
    for (size_t i = 0; i < hm.size(); ++i) hm_sig[i] = sigmoid(hm[i]);
    std::vector<float> hm_nms = nms_heatmap(hm_sig, classes, H, W);

    struct Candidate {
        float score;
        int cls;
        int y;
        int x;
    };

    auto cmp = [](const Candidate& a, const Candidate& b) {
        return a.score > b.score;
    };

    std::vector<Candidate> all_candidates;
    all_candidates.reserve(K * classes);
    for (int c = 0; c < classes; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float s = hm_nms[((c * H) + y) * W + x];
                if (s > CENTER_PEAK_THRESH) {
                    all_candidates.push_back({s, c, y, x});
                }
            }
        }
    }
    if (all_candidates.size() > static_cast<size_t>(K)) {
        std::nth_element(all_candidates.begin(), all_candidates.begin() + K, all_candidates.end(), cmp);
        all_candidates.resize(K);
    }
    std::sort(all_candidates.begin(), all_candidates.end(), cmp);

    std::vector<DecodeResult> results;
    results.reserve(all_candidates.size());
    for (const auto& cand : all_candidates) {
        int c = cand.cls;
        int y = cand.y;
        int x = cand.x;

        float off_x = offset[(0 * H + y) * W + x];
        float off_y = offset[(1 * H + y) * W + x];
        float dir_x = direction[(0 * H + y) * W + x];
        float dir_y = direction[(1 * H + y) * W + x];
        float z = zcoor[(0 * H + y) * W + x] * MAX_Z_M;
        float h = dim[(0 * H + y) * W + x] * DIM_SCALE;
        float w = dim[(1 * H + y) * W + x] * DIM_SCALE;
        float l = dim[(2 * H + y) * W + x] * DIM_SCALE;

        DecodeResult det;
        det.score = cand.score;
        det.x = static_cast<float>(x) + off_x;
        det.y = static_cast<float>(y) + off_y;
        det.z = z;
        det.h = h;
        det.w = w;
        det.l = l;
        det.dirx = dir_x;
        det.diry = dir_y;
        det.cls = static_cast<float>(c);
        results.push_back(det);
    }
    return results;
}

std::array<std::vector<ProcessedDetection>, NUM_CLASSES> post_processing(
    const std::vector<DecodeResult>& dets) {
    std::array<std::vector<ProcessedDetection>, NUM_CLASSES> per_class;
    for (const auto& det : dets) {
        int cls = static_cast<int>(det.cls);
        if (cls < 0 || cls >= NUM_CLASSES) continue;
        if (det.score <= CENTER_PEAK_THRESH) continue;

        ProcessedDetection pd;
        pd.score = det.score;
        pd.x_px = det.x * DOWN_RATIO;
        pd.y_px = det.y * DOWN_RATIO;
        pd.z_m = det.z;
        pd.h_m = det.h;
        pd.w_px = (det.w / 100.0f) * GRID;
        pd.l_px = (det.l / 100.0f) * GRID;
        pd.yaw = std::atan2(det.dirx, det.diry);
        pd.cls = cls;
        per_class[cls].push_back(pd);
    }
    return per_class;
}

float iou_axis_aligned(const ProcessedDetection& a, const ProcessedDetection& b) {
    float ax1 = a.x_px - a.w_px * 0.5f;
    float ay1 = a.y_px - a.l_px * 0.5f;
    float ax2 = a.x_px + a.w_px * 0.5f;
    float ay2 = a.y_px + a.l_px * 0.5f;

    float bx1 = b.x_px - b.w_px * 0.5f;
    float by1 = b.y_px - b.l_px * 0.5f;
    float bx2 = b.x_px + b.w_px * 0.5f;
    float by2 = b.y_px + b.l_px * 0.5f;

    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1);
    float area_b = std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1);
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

std::vector<ProcessedDetection> nms_single_class(const std::vector<ProcessedDetection>& dets, float nms_thresh) {
    if (dets.empty()) return {};
    std::vector<ProcessedDetection> sorted = dets;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });

    std::vector<ProcessedDetection> kept;
    for (const auto& det : sorted) {
        bool suppress = false;
        for (const auto& k : kept) {
            if (iou_axis_aligned(det, k) > nms_thresh) {
                suppress = true;
                break;
            }
        }
        if (!suppress) {
            kept.push_back(det);
        }
    }
    return kept;
}

void apply_nms(std::array<std::vector<ProcessedDetection>, NUM_CLASSES>& per_class,
               const std::array<float, NUM_CLASSES>& nms_thresh_by_class) {
    for (int cls = 0; cls < NUM_CLASSES; ++cls) {
        float thr = nms_thresh_by_class[cls];
        if (thr <= 0.0f) continue;
        per_class[cls] = nms_single_class(per_class[cls], thr);
    }
}

std::pair<double, double> px_to_ego(double v_px, double u_px, const simpletrack::Boundary& boundary) {
    const double d = boundary.discretization();
    const double H = GRID + 1;
    const double W = GRID + 1;
    double x_e = (v_px + 0.5 - H / 4.0) * d;
    double y_img = (u_px + 0.5 - W / 2.0) * d;
    double y_e = -y_img;
    if (g_debug_coords) {
        static int coord_log_count = 0;
        if (coord_log_count < 10) {
            std::cout << "[px_to_ego] v=" << v_px << " u=" << u_px
                      << " -> (x=" << x_e << ", y=" << y_e << ")\n";
            coord_log_count++;
        }
    }
    return {x_e, y_e};
}

std::pair<int, int> ego_to_px(double x_e, double y_e, const simpletrack::Boundary& boundary) {
    const double d = boundary.discretization();
    const double H = GRID + 1;
    const double W = GRID + 1;
    double v = (x_e / d) + H / 4.0 - 0.5;
    double u = ((-y_e) / d) + W / 2.0 - 0.5;
    return {static_cast<int>(std::round(v)), static_cast<int>(std::round(u))};
}

std::vector<Box> detections_to_boxes(const std::array<std::vector<ProcessedDetection>, NUM_CLASSES>& per_class,
                                     const simpletrack::Boundary& boundary) {
    std::vector<Box> boxes;
    size_t total = 0;
    for (const auto& cls_dets : per_class) total += cls_dets.size();
    boxes.reserve(total);
    for (int cls = 0; cls < NUM_CLASSES; ++cls) {
        for (const auto& det : per_class[cls]) {
            auto [x_e, y_e] = px_to_ego(det.y_px, det.x_px, boundary);
            Box box{};
            box.x = static_cast<float>(x_e);
            box.y = static_cast<float>(y_e);
            box.w = static_cast<float>(det.w_px * boundary.discretization());
            box.h = static_cast<float>(det.l_px * boundary.discretization());
            box.angle = det.yaw;
            box.confidence = det.score;
            box.class_type = static_cast<float>(cls);
            boxes.push_back(box);
        }
    }
    return boxes;
}

simpletrack::FrameData detections_to_framedata(
    const std::vector<ProcessedDetection>& detections,
    double timestamp,
    const Eigen::Matrix4d& ego,
    const std::string& cls_name,
    const simpletrack::Boundary& boundary) {

    std::vector<simpletrack::RawDetection> raws;
    raws.reserve(detections.size());
    for (const auto& det : detections) {
        auto [x_e, y_e] = px_to_ego(det.y_px, det.x_px, boundary);
        simpletrack::RawDetection raw;
        raw.score = det.score;
        raw.x_px = static_cast<float>(det.x_px);
        raw.y_px = static_cast<float>(det.y_px);
        raw.z_m = det.z_m;
        raw.h_m = det.h_m;
        raw.w_px = det.w_px;
        raw.l_px = det.l_px;
        raw.yaw_rad = det.yaw;
        raw.cls = det.cls;
        raws.push_back(raw);
        if (g_debug_det) {
            std::cout << "[detections_to_framedata] cls=" << det.cls
                      << " yaw_px=" << det.yaw
                      << " -> (x_e=" << x_e << ", y_e=" << y_e << ")\n";
        }
    }
    return simpletrack::detections_to_framedata(raws, timestamp, ego, true, cls_name, boundary);
}

simpletrack::TrackerConfig load_tracker_config(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    simpletrack::TrackerConfig cfg;
    auto running = root["running"];
    if (running) {
        if (running["match_type"]) cfg.running.match_type = running["match_type"].as<std::string>();
        if (running["asso"]) cfg.running.asso = running["asso"].as<std::string>();
        if (running["score_threshold"]) cfg.running.score_threshold = running["score_threshold"].as<double>();
        if (running["max_age_since_update"]) cfg.running.max_age_since_update = running["max_age_since_update"].as<int>();
        if (running["min_hits_to_birth"]) cfg.running.min_hits_to_birth = running["min_hits_to_birth"].as<int>();
        if (running["motion_model"]) cfg.running.motion_model = running["motion_model"].as<std::string>();
        if (running["asso_thres"]) {
            for (const auto& kv : running["asso_thres"]) {
                cfg.running.asso_thresholds[kv.first.as<std::string>()] = kv.second.as<double>();
            }
        }
    }
    auto redundancy = root["redundancy"];
    if (redundancy) {
        if (redundancy["mode"]) cfg.redundancy.mode = redundancy["mode"].as<std::string>();
        if (redundancy["det_score_threshold"]) {
            for (const auto& kv : redundancy["det_score_threshold"]) {
                cfg.redundancy.det_score_threshold[kv.first.as<std::string>()] = kv.second.as<double>();
            }
        }
        if (redundancy["det_dist_threshold"]) {
            for (const auto& kv : redundancy["det_dist_threshold"]) {
                cfg.redundancy.det_dist_threshold[kv.first.as<std::string>()] = kv.second.as<double>();
            }
        }
    }
    return cfg;
}

simpletrack::TrackerConfig make_cfg(const simpletrack::TrackerConfig& base,
                                    double giou_asso,
                                    double giou_redund,
                                    int max_age,
                                    int min_hits,
                                    double score_thr,
                                    double nms_th,
                                    const std::vector<double>& measurement_noise,
                                    std::optional<double> post_nms_iou) {
    simpletrack::TrackerConfig cfg = base;
    cfg.running.asso = "giou";
    cfg.running.asso_thresholds["giou"] = giou_asso;
    cfg.redundancy.det_dist_threshold["giou"] = giou_redund;
    cfg.running.max_age_since_update = max_age;
    cfg.running.min_hits_to_birth = min_hits;
    cfg.running.score_threshold = score_thr;
    cfg.running.post_nms_iou = post_nms_iou.value_or(0.0);
    if (!measurement_noise.empty()) {
        Eigen::Matrix<double, 7, 7> R = Eigen::Matrix<double, 7, 7>::Zero();
        for (size_t i = 0; i < measurement_noise.size() && i < 7; ++i) {
            R(i, i) = measurement_noise[i];
        }
        cfg.running.measurement_noise = R;
    } else {
        cfg.running.measurement_noise.reset();
    }
    (void)nms_th;
    return cfg;
}

struct ClassParams {
    std::string name;
    double giou_asso = 1.0;
    double giou_redund = 0.5;
    int max_age = 3;
    int min_hits = 3;
    double score_thr = 0.5;
    double nms_th = 0.5;
    double post_nms_iou = 0.0;
    std::vector<double> measurement_noise;
};

std::unordered_map<std::string, ClassParams> load_class_params(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    YAML::Node classes = root["classes"];
    if (!classes) {
        throw std::runtime_error("Class config YAML missing 'classes' node: " + path);
    }
    std::unordered_map<std::string, ClassParams> params;
    for (auto it = classes.begin(); it != classes.end(); ++it) {
        const std::string key = it->first.as<std::string>();
        const YAML::Node& node = it->second;
        ClassParams cp;
        if (node["name"]) cp.name = node["name"].as<std::string>();
        if (node["giou_asso"]) cp.giou_asso = node["giou_asso"].as<double>();
        if (node["giou_redund"]) cp.giou_redund = node["giou_redund"].as<double>();
        if (node["max_age"]) cp.max_age = node["max_age"].as<int>();
        if (node["min_hits"]) cp.min_hits = node["min_hits"].as<int>();
        if (node["score_thr"]) cp.score_thr = node["score_thr"].as<double>();
        if (node["nms_th"]) cp.nms_th = node["nms_th"].as<double>();
        if (node["post_nms_iou"]) cp.post_nms_iou = node["post_nms_iou"].as<double>();
        if (node["measurement_noise"]) cp.measurement_noise = node["measurement_noise"].as<std::vector<double>>();
        params.emplace(key, cp);
    }
    if (params.empty()) {
        throw std::runtime_error("No class entries found in " + path);
    }
    return params;
}

cv::Mat make_canvas_from_bev(const std::vector<uint8_t>& bev) {
    cv::Mat canvas(GRID, GRID, CV_8UC3);
    for (int y = 0; y < GRID; ++y) {
        for (int x = 0; x < GRID; ++x) {
            int idx = (y * GRID + x) * 3;
            uint8_t b = bev[idx + 0];
            uint8_t g = bev[idx + 1];
            uint8_t r = bev[idx + 2];
            canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
        }
    }
    cv::Mat canvas_bgr;
    cv::cvtColor(canvas, canvas_bgr, cv::COLOR_RGB2BGR);
    return canvas_bgr;
}

std::string resolve_path(const std::string& path) {
    std::filesystem::path p(path);
    if (std::filesystem::exists(p)) return p.string();
    std::filesystem::path alt = std::filesystem::path("..") / p;
    if (std::filesystem::exists(alt)) return alt.string();
    return path;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        if (!parse_args(argc, argv, args)) {
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "[Args] " << e.what() << "\n";
        return 1;
    }
    args.config_path = resolve_path(args.config_path);
    args.class_config_path = resolve_path(args.class_config_path);

    g_debug_coords = args.debug_coords;
    g_debug_det = args.debug_det;
    g_debug_yaw = args.debug_yaw;

    std::cout << "Initializing Hailo inference..." << std::endl;

    auto hailo = HailoIF::getInstance();
    simpletrack::Boundary boundary = simpletrack::kDefaultBoundary;

    auto bin_files = resolve_bin_inputs(args.data_glob);
    if (bin_files.empty()) {
        // fallback to ../ for relative defaults
        bin_files = resolve_bin_inputs("../" + args.data_glob);
    }
    // Apply start/end frame slicing (0-based, inclusive)
    if (args.start_frame < 0) args.start_frame = 0;
    size_t start_idx = static_cast<size_t>(args.start_frame);
    size_t end_idx = bin_files.empty() ? 0 : bin_files.size() - 1;
    if (args.end_frame) {
        if (*args.end_frame >= 0) {
            end_idx = static_cast<size_t>(*args.end_frame);
            if (end_idx >= bin_files.size()) {
                end_idx = bin_files.size() - 1;
            }
        }
    }
    if (start_idx >= bin_files.size() || start_idx > end_idx) {
        bin_files.clear();
    } else {
        bin_files = std::vector<std::string>(bin_files.begin() + start_idx,
                                             bin_files.begin() + end_idx + 1);
    }

    if (args.max_frames) {
        if (*args.max_frames < 0) {
            bin_files.clear();
        } else if (static_cast<size_t>(*args.max_frames) < bin_files.size()) {
            bin_files.resize(*args.max_frames);
        }
    }
    if (bin_files.empty()) {
        std::cerr << "[DATA] No .bin files for pattern: " << args.data_glob << "\n";
        return 1;
    }
    std::cout << "[DATA] Loaded " << bin_files.size() << " frames from '" << args.data_glob << "'\n";

    if (!std::filesystem::exists(args.config_path)) {
        std::cerr << "[Config] Tracker config not found: " << args.config_path << "\n";
        return 1;
    }
    if (!std::filesystem::exists(args.class_config_path)) {
        std::cerr << "[Config] Class config not found: " << args.class_config_path << "\n";
        return 1;
    }

    simpletrack::TrackerConfig base_cfg = load_tracker_config(args.config_path);
    auto class_params = load_class_params(args.class_config_path);
    auto fetch_params = [&](const std::string& key) -> const ClassParams& {
        auto it = class_params.find(key);
        if (it == class_params.end()) {
            throw std::runtime_error("Missing class key '" + key + "' in " + args.class_config_path);
        }
        return it->second;
    };

    const ClassParams& veh_params = fetch_params("veh");
    const ClassParams& ped_params = fetch_params("ped");
    const ClassParams& cyc_params = fetch_params("cyc");

    auto cfg_veh = make_cfg(base_cfg, veh_params.giou_asso, veh_params.giou_redund,
                            veh_params.max_age, veh_params.min_hits, veh_params.score_thr,
                            veh_params.nms_th, veh_params.measurement_noise, veh_params.post_nms_iou);
    auto cfg_cyc = make_cfg(base_cfg, cyc_params.giou_asso, cyc_params.giou_redund,
                            cyc_params.max_age, cyc_params.min_hits, cyc_params.score_thr,
                            cyc_params.nms_th, cyc_params.measurement_noise, cyc_params.post_nms_iou);
    auto cfg_ped = make_cfg(base_cfg, ped_params.giou_asso, ped_params.giou_redund,
                            ped_params.max_age, ped_params.min_hits, ped_params.score_thr,
                            ped_params.nms_th, ped_params.measurement_noise, ped_params.post_nms_iou);
    std::array<float, NUM_CLASSES> det_nms_thresh = {
        static_cast<float>(ped_params.nms_th),
        static_cast<float>(veh_params.nms_th),
        static_cast<float>(cyc_params.nms_th)
    };

    simpletrack::MOTModel tracker_veh(cfg_veh);
    simpletrack::MOTModel tracker_cyc(cfg_cyc);
    simpletrack::MOTModel tracker_ped(cfg_ped);

    bool assoc_debug = args.debug_all || args.debug_assoc;
    bool life_debug = args.debug_all || args.debug_life;
    tracker_veh.set_debug(assoc_debug, life_debug);
    tracker_cyc.set_debug(assoc_debug, life_debug);
    tracker_ped.set_debug(assoc_debug, life_debug);

    bool show_gui = !args.no_gui;
    bool pause_for_key = !args.play;
    const std::string win_name = "Tracking";
    if (show_gui) {
        cv::namedWindow(win_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(win_name, 900, 900);
    }

    cv::VideoWriter writer;
    bool writer_initialized = false;
    int frame_period_ms = std::max(1, static_cast<int>(std::round(1000.0 / args.video_fps)));

    Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();

    size_t processed_frames = 0;
    TimingStats preprocess_stats;
    TimingStats load_stats;
    TimingStats filter_stats;
    TimingStats bev_stats;
    TimingStats inference_stats;
    TimingStats parallel_stats;
    TimingStats decode_stats;
    TimingStats tracking_stats;
    TimingStats total_frame_stats;
    TimingStats track_predict_stats;
    TimingStats track_assoc_stats;
    TimingStats track_dist_stats;
    TimingStats track_update_stats;

    for (size_t frame_idx = 0; frame_idx < bin_files.size(); ++frame_idx) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        std::cout << "\n[Frame " << frame_idx << "] " << bin_files[frame_idx] << std::endl;

        auto load_start = std::chrono::high_resolution_clock::now();
        auto raw_cloud = load_point_cloud(bin_files[frame_idx]);
        auto load_after = std::chrono::high_resolution_clock::now();
        auto filtered = remove_points(raw_cloud, boundary);
        auto filter_after = std::chrono::high_resolution_clock::now();
        auto bev_input = make_bev_map_int8(filtered, boundary);
        auto load_end = std::chrono::high_resolution_clock::now();
        auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_after - load_start).count();
        auto filter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(filter_after - load_after).count();
        auto bev_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - filter_after).count();
        auto prep_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
        std::cout << "Preprocess (INT8 BEV " << GRID << "x" << GRID << "x3 NHWC): " << prep_ms
                  << " ms | load=" << load_ms << " ms filter=" << filter_ms << " ms bev=" << bev_ms << " ms" << std::endl;

        update_stats(preprocess_stats, static_cast<double>(prep_ms));
        update_stats(load_stats, static_cast<double>(load_ms));
        update_stats(filter_stats, static_cast<double>(filter_ms));
        update_stats(bev_stats, static_cast<double>(bev_ms));

        std::map<std::string, std::vector<float32_t>> output_data;
        auto infer_start = std::chrono::high_resolution_clock::now();
        auto status = hailo->infer(bev_input, output_data);
        auto infer_end = std::chrono::high_resolution_clock::now();
        auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();

        if (status != HAILO_SUCCESS) {
            std::cerr << "Inference failed with status: " << status << std::endl;
            continue;
        }
        std::cout << "Hailo infer: " << infer_ms << " ms" << std::endl;
        update_stats(inference_stats, static_cast<double>(infer_ms));

        std::vector<float> hm_cen;
        std::vector<float> cen_offset;
        std::vector<float> direction;
        std::vector<float> z_coor;
        std::vector<float> dim;

        auto fpn_start = std::chrono::high_resolution_clock::now();
        hailo->parallelProcess(output_data, hm_cen, cen_offset, direction, z_coor, dim, HEATMAP_H, HEATMAP_W);
        auto fpn_end = std::chrono::high_resolution_clock::now();
        auto fpn_ms = std::chrono::duration_cast<std::chrono::milliseconds>(fpn_end - fpn_start).count();
        std::cout << "Parallel head fusion: " << fpn_ms << " ms" << std::endl;
        update_stats(parallel_stats, static_cast<double>(fpn_ms));

        auto decode_start = std::chrono::high_resolution_clock::now();
        auto decoded = decode(hm_cen, cen_offset, direction, z_coor, dim, NUM_CLASSES, HEATMAP_H, HEATMAP_W, TOPK);
        if (args.debug_all) {
            constexpr float kSmallDirNorm = 1e-3f;
            constexpr float kYawBoundary = static_cast<float>(M_PI) - 0.1f;
            for (size_t i = 0; i < decoded.size(); ++i) {
                float norm = std::sqrt(decoded[i].dirx * decoded[i].dirx + decoded[i].diry * decoded[i].diry);
                float yaw = std::atan2(decoded[i].dirx, decoded[i].diry);
                if (norm < kSmallDirNorm || std::fabs(yaw) > kYawBoundary) {
                    std::cout << "[DirCheck] idx=" << i
                              << " norm=" << norm
                              << " yaw=" << yaw
                              << " cls=" << decoded[i].cls
                              << " dir=(" << decoded[i].dirx << "," << decoded[i].diry << ")\n";
                }
            }
        }
        auto per_class = post_processing(decoded);
        if (g_debug_yaw) {
            size_t total_raw = 0;
            for (const auto& cls_dets : per_class) total_raw += cls_dets.size();
            std::cout << "[Pre-NMS] raw detections: " << total_raw << " (by class: "
                      << per_class[0].size() << ", " << per_class[1].size() << ", " << per_class[2].size() << ")\n";
            for (int cls = 0; cls < NUM_CLASSES; ++cls) {
                for (size_t i = 0; i < per_class[cls].size(); ++i) {
                    const auto& d = per_class[cls][i];
                    std::cout << "  cls=" << cls
                              << " idx=" << i
                              << " score=" << d.score
                              << " x_px=" << d.x_px
                              << " y_px=" << d.y_px
                              << " w_px=" << d.w_px
                              << " l_px=" << d.l_px
                              << " yaw=" << d.yaw << "\n";
                }
            }
        }
        apply_nms(per_class, det_nms_thresh);
        auto boxes = detections_to_boxes(per_class, boundary);
        auto decode_end = std::chrono::high_resolution_clock::now();
        auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start).count();
        std::cout << "Decode/post-process: " << decode_ms << " ms" << std::endl;
        update_stats(decode_stats, static_cast<double>(decode_ms));

        if (args.debug_all) {
            size_t total_dets = 0;
            for (const auto& cls_dets : per_class) total_dets += cls_dets.size();
            std::cout << "Detections -> total: " << total_dets
                      << " | class0: " << per_class[0].size()
                      << " class1: " << per_class[1].size()
                      << " class2: " << per_class[2].size() << std::endl;

            for (size_t i = 0; i < boxes.size(); ++i) {
                const auto& det = boxes[i];
                std::cout << "  idx=" << i
                          << " cls=" << det.class_type
                          << " score=" << det.confidence
                          << " pos_m=(" << det.x << ", " << det.y << ")"
                          << " size_m=(" << det.w << ", " << det.h << ")"
                          << " yaw=" << det.angle << " rad" << std::endl;
            }
        }

        double ts = static_cast<double>(frame_idx) * DT;
        auto fd_ped = detections_to_framedata(per_class[0], ts, I4, "Pedestrian", boundary);
        auto fd_veh = detections_to_framedata(per_class[1], ts, I4, "Car", boundary);
        auto fd_cyc = detections_to_framedata(per_class[2], ts, I4, "Cyclist", boundary);

        auto track_start = std::chrono::high_resolution_clock::now();
        simpletrack::TrackTiming tt_veh, tt_ped, tt_cyc;
        auto res_veh = tracker_veh.frame_mot(fd_veh, &tt_veh);
        auto res_ped = tracker_ped.frame_mot(fd_ped, &tt_ped);
        auto res_cyc = tracker_cyc.frame_mot(fd_cyc, &tt_cyc);
        std::vector<simpletrack::TrackResult> tracks;
        tracks.reserve(res_veh.size() + res_ped.size() + res_cyc.size());
        auto append_nonzero = [&](const std::vector<simpletrack::TrackResult>& src) {
            for (const auto& t : src) {
                if (!std::isnan(t.bbox.s) && t.bbox.s <= 0.0) continue;  // drop zero-score tracks
                tracks.push_back(t);
            }
        };
        append_nonzero(res_veh);
        append_nonzero(res_ped);
        append_nonzero(res_cyc);
        auto track_end = std::chrono::high_resolution_clock::now();
        auto track_ms = std::chrono::duration_cast<std::chrono::milliseconds>(track_end - track_start).count();
        std::cout << "Tracking: " << track_ms << " ms" << std::endl;
        update_stats(track_predict_stats, tt_veh.predict_ms + tt_ped.predict_ms + tt_cyc.predict_ms);
        update_stats(track_assoc_stats, tt_veh.assoc_ms + tt_ped.assoc_ms + tt_cyc.assoc_ms);
        update_stats(track_dist_stats, tt_veh.dist_ms + tt_ped.dist_ms + tt_cyc.dist_ms);
        update_stats(track_update_stats, tt_veh.update_ms + tt_ped.update_ms + tt_cyc.update_ms);
        update_stats(tracking_stats, static_cast<double>(track_ms));

        auto pre_viz_end = std::chrono::high_resolution_clock::now();
        auto frame_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pre_viz_end - frame_start).count();
        std::cout << "Total frame time (preprocess+infer+post+track): " << frame_ms << " ms" << std::endl;
        update_stats(total_frame_stats, static_cast<double>(frame_ms));
        processed_frames += 1;

        cv::Mat canvas = make_canvas_from_bev(bev_input);
        if (args.vis_detections) {
            for (int cls = 0; cls < NUM_CLASSES; ++cls) {
                const cv::Scalar color = COLORS[cls % COLORS.size()];
                for (const auto& det : per_class[cls]) {
                    draw_rotated_box(canvas,
                                     static_cast<float>(det.x_px),
                                     static_cast<float>(det.y_px),
                                     static_cast<float>(det.w_px),
                                     static_cast<float>(det.l_px),
                                     det.yaw,
                                     color,
                                     2);
                }
            }
        }

        struct TrackLabel {
            int u;
            int v;
            int id;
            cv::Scalar color;
        };
        std::vector<TrackLabel> labels;

        if (args.vis_tracks) {
            for (const auto& track : tracks) {
                auto [v, u] = ego_to_px(track.bbox.x, track.bbox.y, boundary);
                if (u < 0 || u >= GRID || v < 0 || v >= GRID) continue;
                float w_px = static_cast<float>(track.bbox.w / boundary.discretization());
                float l_px = static_cast<float>(track.bbox.l / boundary.discretization());
                float yaw_draw = -static_cast<float>(track.bbox.o);
                const cv::Scalar& color = COLORS[track.id % COLORS.size()];
                draw_rotated_box(canvas,
                                 static_cast<float>(u),
                                 static_cast<float>(v),
                                 w_px,
                                 l_px,
                                 -yaw_draw,
                                 color,
                                  2);
                labels.push_back(TrackLabel{u, v, track.id, color});
            }
        }

        cv::Mat canvas_show;
        cv::rotate(canvas, canvas_show, cv::ROTATE_180);
        if (args.vis_tracks) {
            for (const auto& lbl : labels) {
                int u_rot = GRID - 1 - lbl.u;
                int v_rot = GRID - 1 - lbl.v;
                std::string label = "ID:" + std::to_string(lbl.id);
                int text_x = u_rot;
                int text_y = v_rot - 5;
                cv::putText(canvas_show, label, cv::Point(text_x, text_y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            }
        }
        if (!args.video_path.empty() && !writer_initialized) {
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            writer.open(args.video_path, fourcc, args.video_fps, canvas_show.size());
            if (!writer.isOpened()) {
                std::cerr << "Failed to open video writer at " << args.video_path << std::endl;
            } else {
                writer_initialized = true;
                std::cout << "[Video] Recording to " << args.video_path
                          << " @ " << std::fixed << std::setprecision(2) << args.video_fps << " FPS\n";
            }
        }
        if (writer_initialized) {
            writer.write(canvas_show);
        }
        if (cv::imwrite("result.png", canvas_show)) {
            std::cout << "Saved visualization -> result.png" << std::endl;
        } else {
            std::cerr << "Failed to save result.png" << std::endl;
        }

        if (show_gui) {
            cv::imshow(win_name, canvas_show);
            int wait_ms = pause_for_key ? 0 : frame_period_ms;
            int key = cv::waitKey(wait_ms) & 0xFF;
            if (key == 'q' || key == 27) {
                std::cout << "[INFO] Early termination requested.\n";
                break;
            }
        } else {
            if ((frame_idx % 10 == 0) || (frame_idx + 1 == bin_files.size())) {
                std::cout << "[Progress] frame " << (frame_idx + 1) << "/" << bin_files.size() << "\n";
            }
        }
    }

    if (writer.isOpened()) {
        writer.release();
        std::cout << "Saved video -> " << args.video_path << std::endl;
    }
    if (show_gui) cv::destroyAllWindows();

    if (processed_frames > 0) {
        auto print_stats = [&](const std::string& label, const TimingStats& stats, bool show_details = false) {
            double avg = stats.sum / processed_frames;
            double min_v = (stats.min == std::numeric_limits<double>::max()) ? 0.0 : stats.min;
            double max_v = stats.max;
            std::cout << label << avg << " ms"
                      << " (min=" << min_v << " max=" << max_v << ")\n";
        };

        std::cout << "\n=== AVERAGE TIMINGS OVER " << processed_frames << " FRAMES ===\n";
        print_stats("Preprocess total: ", preprocess_stats);
        print_stats("  - load ", load_stats);
        print_stats("  - filter ", filter_stats);
        print_stats("  - bev ", bev_stats);
        print_stats("Inference: ", inference_stats);
        print_stats("Parallel head fusion: ", parallel_stats);
        print_stats("Decode/post-process: ", decode_stats);
        print_stats("Tracking: ", tracking_stats);
        print_stats("  - predict ", track_predict_stats);
        print_stats("  - assoc ", track_assoc_stats);
        print_stats("  - dist ", track_dist_stats);
        print_stats("  - update ", track_update_stats);
        print_stats("Total frame time (preprocess+infer+post+track): ", total_frame_stats);
    }

    return 0;
}
