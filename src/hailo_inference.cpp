#include "../include/hailo_inference.h"
#include <array>
#include <filesystem>
#include <iostream>
#include <cstdio>

using namespace hailort;

#define INFO(...) printf(__VA_ARGS__); printf("\n")
#define ERROR(...) fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n")



#define CLAMP(X,MIN,MAX) ((X>MAX)? MAX:(X<MIN)? MIN:X)

/////////////////////////////////////////////////////////
// Initialize params
float theta = M_PI * 1 / 2;

////////////////////////////////////////////////////////////

namespace {

std::string resolve_hef_path() {
    const std::array<std::filesystem::path, 2> candidates = {
        std::filesystem::path(kDefaultHefFile),
        std::filesystem::path("..") / kDefaultHefFile,
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }

    return candidates.front().string();
}

}  // namespace

HailoIF::HailoIF()
{
    INFO("create instance");
// Legacy init code kept under #if 0 in original source.

}
HailoIF::~HailoIF()
{
    INFO("remove instance");
}

std::shared_ptr<HailoIF> HailoIF::getInstance()
{
    static std::shared_ptr<HailoIF> instance = std::make_shared<HailoIF>();
    return instance;
}

Expected<std::shared_ptr<ConfiguredNetworkGroup>> HailoIF::configure_network_group(Device& device)
{
    const auto hef_path = resolve_hef_path();
    INFO("loading HEF from %s", hef_path.c_str());

    auto hef = Hef::create(hef_path);

    if(!hef) {
        return make_unexpected(hef.status());
    }

    auto configure_params = hef->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);

    if(!configure_params) {
        return make_unexpected(configure_params.status());
    }

    auto network_groups = device.configure(hef.value(), configure_params.value());

    if(!network_groups) {
        return make_unexpected(network_groups.status());
    }

    if(1 != network_groups->size()) {
        ERROR("Invalid amount of network groups");
        return make_unexpected(HAILO_INTERNAL_FAILURE);
    }

    return std::move(network_groups->at(0));
}
hailo_status HailoIF::infer( std::vector<uint8_t>& bev_features, std::map<std::string, std::vector<float32_t>>& output_data)
{
    const size_t frames_count = 1;
    auto input_vstreams = pipeline.value().get_input_vstreams();
    std::map<std::string, std::vector<uint8_t>> input_data;

    for(const auto& input_vstream : input_vstreams) {
        input_data.emplace(input_vstream.get().name(), bev_features); // here is input data
    }

    std::map<std::string, MemoryView> input_data_mem_views;

    for(const auto& input_vstream : input_vstreams) {
        auto& input_buffer = input_data[input_vstream.get().name()];
        input_data_mem_views.emplace(input_vstream.get().name(), MemoryView(input_buffer.data(), input_buffer.size()));
    }

    auto output_vstreams = pipeline.value().get_output_vstreams();

    for(const auto& output_vstream : output_vstreams) {
        output_data.emplace(output_vstream.get().name(), std::vector<float32_t>(output_vstream.get().get_frame_size() * frames_count));
    }

    std::map<std::string, MemoryView> output_data_mem_views;

    for(const auto& output_vstream : output_vstreams) {
        auto& output_buffer = output_data[output_vstream.get().name()];
        output_data_mem_views.emplace(output_vstream.get().name(), MemoryView(output_buffer.data(), output_buffer.size()));
    }

    hailo_status status = pipeline.value().infer(input_data_mem_views, output_data_mem_views, frames_count);
    return status;
}

float HailoIF::calculate_intersection_over_union(const Box& box1, const Box& box2)
{
    // Calculate the intersection area
    float x1_min = std::max(box1.x - box1.w / 2, box2.x - box2.w / 2);
    float y1_min = std::max(box1.y - box1.h / 2, box2.y - box2.h / 2);
    float x1_max = std::min(box1.x + box1.w / 2, box2.x + box2.w / 2);
    float y1_max = std::min(box1.y + box1.h / 2, box2.y + box2.h / 2);
    float intersection_area = std::max(0.0f, x1_max - x1_min) * std::max(0.0f, y1_max - y1_min);
    // Calculate the union area
    float box1_area = box1.w * box1.h;
    float box2_area = box2.w * box2.h;
    float union_area = box1_area + box2_area - intersection_area;
    // Calculate IoU
    float iou = intersection_area / union_area;
    return iou;
}

std::vector<Box> HailoIF::rotated_nms(const std::vector<std::vector<float>>& bboxes, float nms_thres)
{
    std::vector<Box> predictions;

    for(const auto& bbox : bboxes) {
        Box box;
        box.x = bbox[1];
        box.y = bbox[2];
        box.w = bbox[5];
        box.h = bbox[6];
        box.angle = bbox[7];
        box.confidence = bbox[8];
        box.class_type = bbox[0];

        if (box.w * box.h > 4) predictions.push_back(box);
    }

    std::vector<Box> selected_predictions;

    while(!predictions.empty()) {
        std::sort(predictions.begin(), predictions.end(),
            [](const Box & a, const Box & b) {
                return a.confidence > b.confidence;
            });
        Box current_box = predictions[0];
        selected_predictions.push_back(current_box);
        predictions.erase(predictions.begin());

        for(auto it = predictions.begin(); it != predictions.end();) {
            float iou = calculate_intersection_over_union(current_box, *it);

            if(iou > nms_thres) {
                it = predictions.erase(it);

            } else {
                ++it;
            }
        }
    }

    return selected_predictions;
}


/// for sfa 3d


float clamp_custom(float val, float min_val, float max_val) {
    return std::max(min_val, std::min(max_val, val));
}

// Optimized processAndApplyKfpn function
std::vector<float> HailoIF::processAndApplyKfpn(const std::map<std::string, std::vector<float>>& output_data_map,
                                                const std::vector<std::string>& head_array,
                                                int height, int width, int channels) {
    int num_heads = head_array.size();
    if (num_heads == 0) {
        throw std::invalid_argument("head_array is empty");
    }

    int size = channels * height * width;
    std::vector<float> weighted_sum(size, 0.0f);
    std::vector<float> softmax_sum(size, 0.0f);

    float* weighted_sum_ptr = weighted_sum.data();
    float* softmax_sum_ptr = softmax_sum.data();

    for (int i = 0; i < num_heads; ++i) {
        const auto& output_data_it = output_data_map.find(head_array[i]);
        if (output_data_it == output_data_map.end()) {
            throw std::invalid_argument("Node " + head_array[i] + " not found in output_data_map");
        }

        const std::vector<float>& output_data = output_data_it->second;
        const float* output_data_ptr = output_data.data();

        for (int idx = 0; idx < size; ++idx) {
            float exp_val = std::exp(output_data_ptr[idx]);
            softmax_sum_ptr[idx] += exp_val;
            weighted_sum_ptr[idx] += exp_val * output_data_ptr[idx];
        }
    }

    for (int idx = 0; idx < size; ++idx) {
        if (softmax_sum_ptr[idx] != 0.0f) {
            weighted_sum_ptr[idx] /= softmax_sum_ptr[idx];
        }
    }

    std::vector<float> reordered_weighted_sum(size);
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                int source_idx = h * width * channels + w * channels + c;
                int target_idx = c * height * width + h * width + w;
                reordered_weighted_sum[target_idx] = weighted_sum[source_idx];
            }
        }
    }

    return reordered_weighted_sum;
}

std::vector<float> HailoIF::applySigmoid(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    // Use the lookup table for sigmoid
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = sigmoid_lookup.get(input[i]);
    }

    return output;
}

// Optimized _topk
std::tuple<std::vector<float>, std::vector<int>, std::vector<int>, std::vector<float>, std::vector<float>> HailoIF::_topk(const std::vector<float>& scores, int K, int height, int width) {
    int total_size = scores.size();
    std::vector<std::pair<float, int>> score_inds(total_size);

    // Store scores with their corresponding indices
    for (int i = 0; i < total_size; ++i) {
        score_inds[i] = std::make_pair(scores[i], i);
    }

    // Sort to get the top K elements
    std::partial_sort(score_inds.begin(), score_inds.begin() + K, score_inds.end(), std::greater<>());

    // Resize output vectors once instead of using multiple push_back
    std::vector<float> topk_scores(K);
    std::vector<int> topk_inds(K);
    std::vector<int> topk_clses(K);
    std::vector<float> topk_ys(K);
    std::vector<float> topk_xs(K);

    for (int k = 0; k < K; ++k) {
        topk_scores[k] = score_inds[k].first;
        int flat_index = score_inds[k].second;
        topk_inds[k] = flat_index % (height * width);
        topk_clses[k] = flat_index / (height * width);
        topk_ys[k] = topk_inds[k] / width;
        topk_xs[k] = topk_inds[k] % width;
    }

    return {topk_scores, topk_inds, topk_clses, topk_ys, topk_xs};
}

// Optimized decode
std::vector<std::vector<float>> HailoIF::decode(const std::vector<float>& hm_cen,
                                                const std::vector<float>& cen_offset,
                                                const std::vector<float>& direction,
                                                const std::vector<float>& z_coor,
                                                const std::vector<float>& dim,
                                                float conf_thresh,
                                                int K) {
    int down_ratio = 4;
    int height = 304;
    int width = 304;

    auto [scores, inds, clses, ys, xs] = _topk(hm_cen, K, height, width);

    std::vector<std::vector<float>> ret_ary;
    ret_ary.reserve(K);  // Reserve space to avoid repeated reallocation

    for (int i = 0; i < K; ++i) {
        if (scores[i] <= conf_thresh) continue;

        int flat_index = inds[i];
        int y = flat_index / width;
        int x = flat_index % width;

        std::vector<float> temp_ary(9, 0.0f);

        // Gather values from cen_offset (2 channels)
        temp_ary[1] = (cen_offset[y * width + x] + xs[i]) * down_ratio;
        temp_ary[2] = (cen_offset[height * width + y * width + x] + ys[i]) * down_ratio;

        // Gather values from direction (2 channels)
        temp_ary[7] = std::atan2(direction[y * width + x], direction[height * width + y * width + x]);

        // Gather values from z_coor (1 channel)
        temp_ary[3] = z_coor[y * width + x] * 5.5;

        // Gather values from dim (3 channels)
        temp_ary[4] = dim[y * width + x] * 10;
        temp_ary[5] = dim[height * width + y * width + x] * 10 / 100 * 1216; // bound_size * BEV_Width
        temp_ary[6] = dim[2 * height * width + y * width + x] * 10 / 100 * 1216;

        temp_ary[0] = clses[i];
        temp_ary[8] = scores[i];

        ret_ary.push_back(temp_ary);
    }

    return ret_ary;
}

void HailoIF::parallelProcess(const std::map<std::string, std::vector<float>>& output_data,
                              std::vector<float>& hm_cen, std::vector<float>& cen_offset,
                              std::vector<float>& direction, std::vector<float>& z_coor,
                              std::vector<float>& dim, int heatmap_height, int heatmap_width) {
    // Run processing in parallel using std::async
    std::future<std::vector<float>> hm_cen_future = std::async(std::launch::async, [&, this]() {
        std::vector<std::string> hm_cen_head = {"sfa/resize8", "sfa/conv42", "sfa/conv53"};
        return processAndApplyKfpn(output_data, hm_cen_head, heatmap_height, heatmap_width, 3);
    });

    std::future<std::vector<float>> cen_offset_future = std::async(std::launch::async, [&, this]() {
        std::vector<std::string> cen_offset_heads = {"sfa/resize4", "sfa/conv38", "sfa/conv49"};
        return processAndApplyKfpn(output_data, cen_offset_heads, heatmap_height, heatmap_width, 2);
    });

    std::future<std::vector<float>> direction_future = std::async(std::launch::async, [&, this]() {
        std::vector<std::string> direction_heads = {"sfa/resize5", "sfa/conv39", "sfa/conv50"};
        return processAndApplyKfpn(output_data, direction_heads, heatmap_height, heatmap_width, 2);
    });

    std::future<std::vector<float>> z_coor_future = std::async(std::launch::async, [&, this]() {
        std::vector<std::string> z_coor_heads = {"sfa/resize6", "sfa/conv40", "sfa/conv51"};
        return processAndApplyKfpn(output_data, z_coor_heads, heatmap_height, heatmap_width, 1);
    });

    std::future<std::vector<float>> dim_future = std::async(std::launch::async, [&, this]() {
        std::vector<std::string> dim_heads = {"sfa/resize7", "sfa/conv41", "sfa/conv52"};
        return processAndApplyKfpn(output_data, dim_heads, heatmap_height, heatmap_width, 3);
    });

    // Wait for all futures and get results
    hm_cen = hm_cen_future.get();
    cen_offset = cen_offset_future.get();
    direction = direction_future.get();
    z_coor = z_coor_future.get();
    dim = dim_future.get();
}
