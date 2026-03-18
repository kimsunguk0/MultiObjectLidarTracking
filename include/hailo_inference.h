#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include "hailo/hailort.hpp"
#include "types.h"
#include <future>
#include <thread>

inline constexpr const char* kDefaultHefFile = "sfa.hef";

float clamp_custom(float val, float min_val, float max_val);

// Precomputed sigmoid lookup table for -10 to 10 range
class SigmoidLookup {
public:
    SigmoidLookup() {
        int table_size = 20000; // resolution of 0.001
        float step = 20.0f / table_size;
        lookup_table.resize(table_size + 1);
        for (int i = 0; i <= table_size; ++i) {
            float x = -10.0f + i * step;
            lookup_table[i] = 1.0f / (1.0f + std::exp(-x));
        }
    }

    float get(float x) const {
        if (x <= -10.0f) return 0.0f + 1e-4f;
        if (x >= 10.0f) return 1.0f - 1e-4f;
        int index = static_cast<int>((x + 10.0f) * 1000);
        return clamp_custom(lookup_table[index], 1e-4f, 1.0f - 1e-4f);
    }

private:
    std::vector<float> lookup_table;
};


class HailoIF{
private :

#define DEVICE_INDEX (0)

    hailort::Expected<std::vector<hailo_pcie_device_info_t>> pcie_dev_info = hailort::Device::scan_pcie();
    hailort::Expected<std::unique_ptr<hailort::Device>> device = hailort::Device::create_pcie((pcie_dev_info.value())[DEVICE_INDEX]);
    hailort::Expected<std::shared_ptr<hailort::ConfiguredNetworkGroup>> network_group = configure_network_group(*device.value());
    hailort::Expected<std::map<std::string, hailo_vstream_params_t>> input_params = network_group.value()->make_input_vstream_params(true, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, 1);
    hailort::Expected<std::map<std::string, hailo_vstream_params_t>> output_params = network_group.value()->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, 1);
    hailort::Expected<std::unique_ptr<hailort::ActivatedNetworkGroup>> activated_network_group = network_group.value()->activate();
    hailort::Expected<hailort::InferVStreams> pipeline = hailort::InferVStreams::create(*network_group.value(), input_params.value(), output_params.value());

public:
    SigmoidLookup sigmoid_lookup;

    HailoIF();
    ~HailoIF();

    static std::shared_ptr<HailoIF> getInstance();
    hailort::Expected<std::shared_ptr<hailort::ConfiguredNetworkGroup>> configure_network_group(hailort::Device& device);
    hailo_status infer( std::vector<uint8_t>& bev_features, std::map<std::string, std::vector<float32_t>>& output_data);
    float calculate_intersection_over_union(const Box& box1, const Box& box2);
    std::vector<Box> rotated_nms(const std::vector<std::vector<float>>& bboxes, float nms_thres);

    std::vector<float> processAndApplyKfpn(const std::map<std::string, std::vector<float>>& output_data_map,
                                           const std::vector<std::string>& head_array,
                                           int height, int width, int channels);
    std::vector<float> applySigmoid(const std::vector<float>& input);
    std::tuple<std::vector<float>, std::vector<int>, std::vector<int>, std::vector<float>, std::vector<float>> _topk(const std::vector<float>& scores, int K, int height, int width);
    std::vector<std::vector<float>> decode(const std::vector<float>& hm_cen,
                                           const std::vector<float>& cen_offset,
                                           const std::vector<float>& direction,
                                           const std::vector<float>& z_coor,
                                           const std::vector<float>& dim,
                                           float conf_thresh,
                                           int K);
    void parallelProcess(const std::map<std::string, std::vector<float>>& output_data,
                         std::vector<float>& hm_cen, std::vector<float>& cen_offset,
                         std::vector<float>& direction, std::vector<float>& z_coor,
                         std::vector<float>& dim, int heatmap_height, int heatmap_width);
};
