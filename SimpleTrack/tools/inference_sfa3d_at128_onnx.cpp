#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <yaml-cpp/yaml.h>

#include "simpletrack/simpletrack.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <glob.h>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>

namespace {

constexpr int GRID = 1216;
constexpr int NUM_CLASSES = 3;
constexpr int DOWN_RATIO = 4;
constexpr float CENTER_PEAK_THRESH = 0.4f;
constexpr float MAX_Z_M = 5.5f;
constexpr float DIM_SCALE = 10.0f;
constexpr float DT = 0.1f;

const std::array<cv::Scalar, 8> COLORS = {
    cv::Scalar(255, 255, 0),  cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),   cv::Scalar(0, 120, 255),
    cv::Scalar(120, 120, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 255, 120), cv::Scalar(255, 0, 120)
};

static bool parse_bool(const std::string& s) {
    std::string t;
    t.reserve(s.size());
    for (char c : s) t.push_back(std::tolower(static_cast<unsigned char>(c)));
    if (t == "1" || t == "true" || t == "yes" || t == "on") return true;
    if (t == "0" || t == "false" || t == "no" || t == "off") return false;
    throw std::runtime_error("Invalid bool: " + s + " (use true/false)");
}

namespace {
bool g_debug_coords = false;
bool g_debug_det = false;
}

struct Args {
    std::string model_path = "/home/a/Downloads/sfa_sim.onnx";
    std::string data_glob = "../at128/*.bin";
    std::string config_path = "../configs/waymo_configs/vc_kf_giou.yaml";
    std::string class_config_path = "./configs/tracker_params.yaml";
    std::string video_path;
    double video_fps = 1.0 / DT;
    bool play = false;
    bool no_gui = false;
    std::optional<int> max_frames;
    bool debug_assoc = false;
    bool debug_life = false;
    bool debug_all = false;
    bool vis_detections = false;
    bool vis_tracks = true;
    bool debug_coords = false;
    bool debug_det = false;
    bool debug_with_hailo = false;  // extra logs to align with Hailo path
};

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for option: " + option);
            }
            return argv[++i];
        };

        if (key == "--model-path") {
            args.model_path = need_value(key);
        } else if (key == "--data-glob") {
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
        } else if (key == "--help" || key == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --model-path <path>       ONNX model path\n"
                      << "  --data-glob <pattern>     LiDAR .bin glob pattern\n"
                      << "  --config-path <path>      Tracker config YAML path\n"
                      << "  --class-config <path>     Class-specific tracker params YAML\n"
                      << "  --video-path <path>       Optional output video path\n"
                      << "  --video-fps <fps>         Output video FPS (default 10)\n"
                      << "  --play                    Disable pause between frames\n"
                      << "  --no-gui                  Run without OpenCV window\n"
                      << "  --max-frames <N>          Limit processed frames\n"
                      << "  --debug-assoc             Enable association debug logs\n"
                      << "  --debug-life              Enable lifecycle debug logs\n"
                      << "  --debug-all               Enable both debug logs\n"
                      << "  --detection-vis <bool>    Draw raw detections (default false)\n"
                      << "  --tracking-vis <bool>     Draw tracking boxes (default true)\n"
                      << "  --debug-coords            Log BEV->ego coordinate conversions\n"
                      << "  --debug-det               Log detections_to_framedata yaw conversions\n"
                      << "  --debug-with-hailo        Extra logs for comparing with Hailo pipeline\n";
            return false;
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }
    return true;
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

struct Point {
    float x;
    float y;
    float z;
    float intensity;
};

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

struct CellInfo {
    float max_z = -std::numeric_limits<float>::infinity();
    float intensity = 0.0f;
    int count = 0;
    bool has_value = false;
};

std::pair<double, double> px_to_ego(double v_px, double u_px, const simpletrack::Boundary& boundary);

std::vector<float> make_bev_map(const std::vector<Point>& cloud, const simpletrack::Boundary& boundary) {
    const int H = GRID + 1;
    const int W = GRID + 1;
    const double d = boundary.discretization();
    static int bev_debug_count = 0;

    std::vector<CellInfo> cells(H * W);

    for (const auto& p : cloud) {
        if (p.z < 0.f || p.z > MAX_Z_M) continue;
        int xi = static_cast<int>(std::floor(p.x / d + H / 4.0));
        int yi = static_cast<int>(std::floor(p.y / d + W / 2.0));
        if (xi < 0 || xi >= H || yi < 0 || yi >= W) continue;

        if (g_debug_coords && bev_debug_count < 20) {
            double v_center = static_cast<double>(xi) + 0.5;
            double u_center = static_cast<double>(yi) + 0.5;
            auto [x_round, y_round] = px_to_ego(v_center, u_center, boundary);
            double py_x = (v_center - H / 4.0) * d;
            double py_y_img = (u_center - W / 2.0) * d;
            double py_y = -py_y_img;
            std::cout << "[bev_map] ego_in=(" << p.x << ", " << p.y << ")"
                      << " -> grid(v=" << xi << ", u=" << yi << ")"
                      << " -> roundtrip(x=" << x_round << ", y=" << y_round << ")"
                      << " | delta=(" << (x_round - p.x) << ", " << (y_round - p.y) << ")"
                      << " | python_est=(" << py_x << ", " << py_y << ")\n";
            ++bev_debug_count;
        }

        CellInfo& cell = cells[xi * W + yi];
        if (!cell.has_value || p.z > cell.max_z) {
            cell.max_z = p.z;
            cell.intensity = p.intensity;
        }
        cell.count += 1;
        cell.has_value = true;
    }

    std::vector<float> density(H * W, 0.0f);
    std::vector<float> height(H * W, 0.0f);
    std::vector<float> intensity(H * W, 0.0f);

    const double z_range = boundary.z_range();
    for (int idx = 0; idx < H * W; ++idx) {
        const auto& cell = cells[idx];
        if (!cell.has_value) continue;
        float h = cell.max_z / static_cast<float>(z_range);
        h = std::clamp(h, 0.0f, 1.0f);
        float inten = std::clamp(cell.intensity, 0.0f, 255.0f);
        float dens = std::min(1.0f, std::log(static_cast<float>(cell.count) + 1.0f) / std::log(128.0f));

        uint8_t h_u8 = static_cast<uint8_t>(h * 255.0f);
        uint8_t i_u8 = static_cast<uint8_t>(inten);
        uint8_t d_u8 = static_cast<uint8_t>(dens * 255.0f);

        height[idx] = static_cast<float>(h_u8) / 255.0f;
        intensity[idx] = static_cast<float>(i_u8) / 255.0f;
        density[idx] = static_cast<float>(d_u8) / 255.0f;
    }

    std::vector<float> bev(3 * GRID * GRID, 0.0f);
    for (int y = 0; y < GRID; ++y) {
        for (int x = 0; x < GRID; ++x) {
            int src_idx = y * W + x;
            int dst_idx = y * GRID + x;
            bev[0 * GRID * GRID + dst_idx] = intensity[src_idx];
            bev[1 * GRID * GRID + dst_idx] = height[src_idx];
            bev[2 * GRID * GRID + dst_idx] = density[src_idx];
        }
    }
    return bev;
}

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
    // std::cout << "[decode] sample yaw entries:\n";
    // for (size_t idx = 0; idx < std::min<size_t>(5, results.size()); ++idx) {
    //     const auto& det = results[idx];
    //     std::cout << "  idx=" << idx
    //               << " score=" << det.score
    //               << " x=" << det.x
    //               << " y=" << det.y
    //               << " yaw_raw=" << std::atan2(det.dirx, det.diry)
    //               << "\n";
    // }
    return results;
}

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

cv::Point2f rotate_point(float x, float y, float w, float l, float yaw) {
    float cos_y = std::cos(yaw);
    float sin_y = std::sin(yaw);
    return cv::Point2f(x - w / 2 * cos_y - l / 2 * sin_y,
                       y - w / 2 * sin_y + l / 2 * cos_y);
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

std::vector<Point> to_point_vector(const std::vector<float>& raw) {
    std::vector<Point> cloud;
    cloud.reserve(raw.size() / 4);
    for (size_t i = 0; i + 3 < raw.size(); i += 4) {
        cloud.emplace_back(Point{raw[i], raw[i + 1], raw[i + 2], raw[i + 3]});
    }
    return cloud;
}

std::vector<float> bev_to_bgr(const std::vector<float>& bev) {
    std::vector<float> bgr(3 * GRID * GRID);
    for (int y = 0; y < GRID; ++y) {
        for (int x = 0; x < GRID; ++x) {
            int idx = y * GRID + x;
            float intensity = bev[0 * GRID * GRID + idx];
            float height = bev[1 * GRID * GRID + idx];
            float density = bev[2 * GRID * GRID + idx];
            bgr[idx * 3 + 0] = intensity;
            bgr[idx * 3 + 1] = height;
            bgr[idx * 3 + 2] = density;
        }
    }
    return bgr;
}

int class_token(int cls_idx) {
    switch (cls_idx) {
        case 0: return 2;
        case 1: return 1;
        case 2: return 4;
        default: return 1;
    }
}

struct Stats {
    float min = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
};

Stats compute_stats(const std::vector<float>& v) {
    Stats s;
    if (v.empty()) return s;
    double sum = 0.0;
    s.min = std::numeric_limits<float>::max();
    s.max = std::numeric_limits<float>::lowest();
    for (float x : v) {
        s.min = std::min(s.min, x);
        s.max = std::max(s.max, x);
        sum += x;
    }
    s.mean = static_cast<float>(sum / static_cast<double>(v.size()));
    return s;
}

void log_output_shapes(const std::vector<std::string>& names, const std::vector<std::vector<float>>& tensors) {
    std::cout << "[ONNX] Output tensors:\n";
    for (size_t i = 0; i < names.size() && i < tensors.size(); ++i) {
        std::cout << "  " << names[i] << " size=" << tensors[i].size() << "\n";
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
        double x_e_py = x_e;
        double y_e_py = y_e;
        static int coord_log_count = 0;
        // if (coord_log_count < 20) {
        //     std::cout << "[px_to_ego] v=" << v_px << " u=" << u_px
        //               << " -> current(x=" << x_e << ", y=" << y_e << ")"
        //               << " | python(x=" << x_e_py << ", y=" << y_e_py << ")\n";
        //     coord_log_count++;
        // }
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
        // if (g_debug_det) {
        //     std::cout << "[detections_to_framedata] cls=" << det.cls
        //               << " yaw_px=" << det.yaw
        //               << " -> (x_e=" << x_e << ", y_e=" << y_e << ")\n";
        // }
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

    g_debug_coords = args.debug_coords;
    g_debug_det = args.debug_det;

    auto scene_paths = glob_files(args.data_glob);
    if (scene_paths.empty()) {
        std::cerr << "[DATA] No .bin files for pattern: " << args.data_glob << "\n";
        return 1;
    }
    if (args.max_frames) {
        if (*args.max_frames < 0) {
            scene_paths.clear();
        } else if (static_cast<size_t>(*args.max_frames) < scene_paths.size()) {
            scene_paths.resize(*args.max_frames);
        }
    }
    std::cout << "[DATA] Loaded " << scene_paths.size() << " frames from pattern '" << args.data_glob << "'\n";

    bool pause_for_key = !args.play;
    if (!args.video_path.empty()) pause_for_key = false;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "simpletrack");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    Ort::Session session(env, args.model_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    size_t num_inputs = session.GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, allocator);
        input_names.emplace_back(name.get());
    }
    size_t num_outputs = session.GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(i, allocator);
        output_names.emplace_back(name.get());
    }
    if (input_names.empty()) {
        std::cerr << "[ONNX] No inputs in model\n";
        return 1;
    }
    std::cout << "[ONNX] inputs: " << input_names.front() << " outputs: ";
    for (size_t i = 0; i < output_names.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << output_names[i];
    }
    std::cout << "\n";

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

    simpletrack::MOTModel tracker_veh(cfg_veh);
    simpletrack::MOTModel tracker_cyc(cfg_cyc);
    simpletrack::MOTModel tracker_ped(cfg_ped);

    bool assoc_debug = args.debug_all || args.debug_assoc;
    bool life_debug = args.debug_all || args.debug_life;
    tracker_veh.set_debug(assoc_debug, life_debug);
    tracker_cyc.set_debug(assoc_debug, life_debug);
    tracker_ped.set_debug(assoc_debug, life_debug);

    auto lookup = [](const auto& map, const std::string& key, double fallback = 0.0) -> double {
        auto it = map.find(key);
        return it != map.end() ? it->second : fallback;
    };

    auto print_params = [&](const std::string& tag,
                            const ClassParams& params,
                            const simpletrack::TrackerConfig& cfg) {
        std::cout << "[" << tag << "] "
                  << lookup(cfg.running.asso_thresholds, "giou") << " "
                  << lookup(cfg.redundancy.det_dist_threshold, "giou") << " "
                  << cfg.running.max_age_since_update << "\n";
        std::cout << "      measurement_noise: ";
        for (double v : params.measurement_noise) {
            std::cout << v << " ";
        }
        std::cout << "\n";
    };

    std::cout << "[Tracker] initialized with class-wise configs\n";
    print_params("veh", veh_params, cfg_veh);
    print_params("cyc", cyc_params, cfg_cyc);
    print_params("ped", ped_params, cfg_ped);

    if (assoc_debug || life_debug) {
        std::cout << "[Debug] association=" << std::boolalpha << assoc_debug
                  << " lifecycle=" << life_debug << std::noboolalpha << "\n";
    }

    bool show_gui = !args.no_gui;
    const std::string win_name = "Tracking";
    if (show_gui) {
        cv::namedWindow(win_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(win_name, 900, 900);
    }

    cv::VideoWriter writer;
    if (!args.video_path.empty()) {
        std::string ext = args.video_path.substr(args.video_path.find_last_of('.') + 1);
        int fourcc = (ext == "mp4" || ext == "MP4") ? cv::VideoWriter::fourcc('m', 'p', '4', 'v')
                                                    : cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        writer.open(args.video_path, fourcc, args.video_fps, cv::Size(GRID, GRID));
        if (!writer.isOpened()) {
            std::cerr << "[Video] Failed to open writer at " << args.video_path << "\n";
            return 1;
        }
        std::cout << "[Video] Recording to " << args.video_path
                  << " @ " << std::fixed << std::setprecision(2) << args.video_fps << " FPS\n";
    }

    int frame_period_ms = std::max(1, static_cast<int>(std::round(1000.0 / args.video_fps)));

    simpletrack::Boundary boundary = simpletrack::kDefaultBoundary;
    Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();

    std::vector<float> bev_input_tensor(1 * 3 * GRID * GRID);
    std::array<int64_t, 4> input_shape = {1, 3, GRID, GRID};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (size_t i = 0; i < scene_paths.size(); ++i) {
        auto raw_cloud = load_point_cloud(scene_paths[i]);
        auto filtered = remove_points(raw_cloud, boundary);
        auto bev = make_bev_map(filtered, boundary);
        std::copy(bev.begin(), bev.end(), bev_input_tensor.begin());

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, bev_input_tensor.data(), bev_input_tensor.size(),
            input_shape.data(), input_shape.size());

        std::vector<const char*> input_cstr;
        input_cstr.reserve(input_names.size());
        for (const auto& s : input_names) input_cstr.push_back(s.c_str());

        std::vector<const char*> output_cstr;
        output_cstr.reserve(output_names.size());
        for (const auto& s : output_names) output_cstr.push_back(s.c_str());

        auto outputs = session.Run(Ort::RunOptions{nullptr},
                                   input_cstr.data(), &input_tensor, 1,
                                   output_cstr.data(), output_cstr.size());

        if (outputs.size() < 5) {
            std::cerr << "[ONNX] Expected 5 outputs, got " << outputs.size() << "\n";
            return 1;
        }

        std::vector<float> hm, offset, direction, zcoor, dim;
        auto get_tensor = [&](Ort::Value& val, std::vector<float>& dst) {
            auto info = val.GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            size_t total = 1;
            for (auto s : shape) total *= s;
            dst.resize(total);
            const float* src = val.GetTensorData<float>();
            std::copy(src, src + total, dst.begin());
            return shape;
        };

        auto hm_shape = get_tensor(outputs[0], hm);  // (1,C,H,W)
        get_tensor(outputs[1], offset);              // (1,2,H,W)
        get_tensor(outputs[2], direction);           // (1,2,H,W)
        get_tensor(outputs[3], zcoor);               // (1,1,H,W)
        auto dim_shape = get_tensor(outputs[4], dim);  // (1,3,H,W)

        if (args.debug_with_hailo) {
            std::vector<std::string> names = {
                output_names.size() > 0 ? output_names[0] : "hm",
                output_names.size() > 1 ? output_names[1] : "offset",
                output_names.size() > 2 ? output_names[2] : "direction",
                output_names.size() > 3 ? output_names[3] : "zcoor",
                output_names.size() > 4 ? output_names[4] : "dim"
            };
            std::vector<std::vector<float>> tensors = {hm, offset, direction, zcoor, dim};
            log_output_shapes(names, tensors);
            auto hm_stats = compute_stats(hm);
            float hm_max_sig = 0.0f;
            size_t hm_over_02 = 0;
            size_t hm_over_04 = 0;
            for (float v : hm) {
                float s = sigmoid(v);
                hm_max_sig = std::max(hm_max_sig, s);
                if (s > 0.2f) hm_over_02++;
                if (s > 0.4f) hm_over_04++;
            }
            std::cout << "[Debug] hm raw min/max/mean=" << hm_stats.min << "/"
                      << hm_stats.max << "/" << hm_stats.mean
                      << " | hm sigmoid max=" << hm_max_sig
                      << " count>0.2=" << hm_over_02
                      << " count>0.4=" << hm_over_04 << "\n";
        }

        int H = static_cast<int>(hm_shape[2]);
        int W = static_cast<int>(hm_shape[3]);
        int down_ratio = std::max(1, static_cast<int>(std::round(static_cast<double>(GRID) / H)));
        auto decoded = decode(hm, offset, direction, zcoor, dim, NUM_CLASSES, H, W, 60);
        auto per_class = post_processing(decoded, down_ratio);

        if (args.debug_with_hailo) {
            size_t det_count = decoded.size();
            size_t det_drawn = per_class[0].size() + per_class[1].size() + per_class[2].size();
            std::cout << "[Debug] Decoded dets: raw=" << det_count
                      << " kept_after_thresh=" << det_drawn
                      << " heatmap_size=" << H
                      << " down_ratio=" << down_ratio << "\n";
        }

        double ts = static_cast<double>(i) * DT;

        auto fd_ped = detections_to_framedata(per_class[0], ts, I4, "Pedestrian", boundary);
        auto fd_veh = detections_to_framedata(per_class[1], ts, I4, "Car", boundary);
        auto fd_cyc = detections_to_framedata(per_class[2], ts, I4, "Cyclist", boundary);

        auto res_veh = tracker_veh.frame_mot(fd_veh);
        auto res_ped = tracker_ped.frame_mot(fd_ped);
        auto res_cyc = tracker_cyc.frame_mot(fd_cyc);
        std::vector<simpletrack::TrackResult> tracks;
        tracks.reserve(res_veh.size() + res_ped.size() + res_cyc.size());
        tracks.insert(tracks.end(), res_veh.begin(), res_veh.end());
        tracks.insert(tracks.end(), res_ped.begin(), res_ped.end());
        tracks.insert(tracks.end(), res_cyc.begin(), res_cyc.end());

        cv::Mat canvas(GRID, GRID, CV_8UC3);
        for (int y = 0; y < GRID; ++y) {
            for (int x = 0; x < GRID; ++x) {
                int idx = y * GRID + x;
                uint8_t b = static_cast<uint8_t>(std::clamp(bev[0 * GRID * GRID + idx], 0.0f, 1.0f) * 255.0f);
                uint8_t g = static_cast<uint8_t>(std::clamp(bev[1 * GRID * GRID + idx], 0.0f, 1.0f) * 255.0f);
                uint8_t r = static_cast<uint8_t>(std::clamp(bev[2 * GRID * GRID + idx], 0.0f, 1.0f) * 255.0f);
                canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
            }
        }
        cv::cvtColor(canvas, canvas, cv::COLOR_RGB2BGR);
        std::cout << "-----------------------\n";

        if (args.vis_detections) {
            for (const auto& cls_dets : per_class) {
                for (const auto& det : cls_dets) {
                    draw_rotated_box(canvas,
                                     det.x_px,
                                     det.y_px,
                                     static_cast<float>(det.w_px),
                                     static_cast<float>(det.l_px),
                                     det.yaw,
                                     cv::Scalar(255, 255, 255),
                                     2);
                }
            }
        }

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
            }
        }

        cv::Mat canvas_show;
        cv::rotate(canvas, canvas_show, cv::ROTATE_180);

        if (writer.isOpened()) {
            writer.write(canvas_show);
        }

        if (show_gui) {
#if CV_MAJOR_VERSION >= 4 && CV_MINOR_VERSION >= 5
            try {
                cv::setWindowTitle(win_name, "Tracking " + std::to_string(i + 1));
            } catch (...) {}
#endif
            cv::imshow(win_name, canvas_show);
            int wait_ms = pause_for_key ? 0 : frame_period_ms;
            int key = cv::waitKey(wait_ms) & 0xFF;
            if (key == 'q' || key == 27) {
                std::cout << "[INFO] Early termination requested.\n";
                break;
            }
        } else if (args.video_path.empty()) {
            if (i % 10 == 0 || i + 1 == scene_paths.size()) {
                std::cout << "[Progress] frame " << (i + 1) << "/" << scene_paths.size() << "\n";
            }
        }
    }

    if (writer.isOpened()) {
        writer.release();
        std::cout << "[DONE] Video saved -> " << args.video_path << "\n";
    }
    if (show_gui) cv::destroyAllWindows();
    std::cout << "[DONE] Tracking run\n";
    return 0;
}
