#include "simpletrack/preprocessing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "simpletrack/geometry.hpp"

namespace simpletrack {

bool weird_bbox(const BBox& bbox) {
    return bbox.l <= 0 || bbox.w <= 0 || bbox.h <= 0;
}

std::pair<std::vector<int>, std::vector<int>> nms(
    const std::vector<BBox>& dets,
    const std::vector<int>& inst_types,
    double threshold_low,
    double threshold_high,
    double threshold_yaw) {

    if (dets.size() != inst_types.size()) {
        throw std::invalid_argument("dets and inst_types must have same length");
    }

    BBoxCoarseFilter coarse_filter(100, 100);
    coarse_filter.bboxes2dict(dets);

    std::vector<double> scores(dets.size());
    std::vector<double> yaws(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        scores[i] = dets[i].has_score() ? dets[i].s : 0.0;
        yaws[i] = dets[i].o;
    }

    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });

    std::vector<int> result_indexes;
    std::vector<int> result_types;

    while (!order.empty()) {
        int index = order.front();
        order.erase(order.begin());

        if (weird_bbox(dets[index])) {
            continue;
        }

        auto related_indexes = coarse_filter.related_bboxes(dets[index]);
        std::vector<int> filtered_related;
        filtered_related.reserve(related_indexes.size());
        for (int idx : related_indexes) {
            if (inst_types[idx] == inst_types[index]) {
                filtered_related.push_back(idx);
            }
        }

        std::vector<double> ious(filtered_related.size(), 0.0);
        for (size_t i = 0; i < filtered_related.size(); ++i) {
            ious[i] = iou3d(dets[index], dets[filtered_related[i]]).second;
        }

        std::vector<int> related_inds;
        for (size_t i = 0; i < filtered_related.size(); ++i) {
            if (ious[i] > threshold_low) {
                related_inds.push_back(filtered_related[i]);
            }
        }

        std::vector<int> order_vote;
        for (size_t i = 0; i < filtered_related.size(); ++i) {
            if (ious[i] > threshold_high) {
                order_vote.push_back(filtered_related[i]);
            }
        }

        if (order_vote.size() >= 2) {
            double median_yaw = 0.0;
            if (order_vote.size() <= 2) {
                int best_idx = *std::max_element(order_vote.begin(), order_vote.end(),
                                                 [&](int a, int b) { return scores[a] < scores[b]; });
                median_yaw = yaws[best_idx];
            } else {
                std::vector<double> yaw_values;
                yaw_values.reserve(order_vote.size());
                for (int idx : order_vote) yaw_values.push_back(yaws[idx]);

                if (order_vote.size() % 2 == 0) {
                    yaw_values.push_back(yaws[order_vote.front()]);
                }
                std::nth_element(yaw_values.begin(),
                                 yaw_values.begin() + yaw_values.size()/2,
                                 yaw_values.end());
                median_yaw = yaw_values[yaw_values.size()/2];
            }

            std::vector<int> yaw_vote;
            for (int idx : order_vote) {
                constexpr double two_pi = 2.0 * 3.14159265358979323846;
                double diff = std::fmod(std::abs(yaws[idx] - median_yaw), two_pi);
                if (diff < threshold_yaw) {
                    yaw_vote.push_back(idx);
                }
            }
            order_vote = yaw_vote;

            if (!order_vote.empty()) {
                double vote_score_sum = 0.0;
                for (int idx : order_vote) vote_score_sum += scores[idx];

                std::vector<double> avg_array(7, 0.0);
                for (int idx : order_vote) {
                    auto arr = dets[idx].to_array_with_score();
                    for (int j = 0; j < 7; ++j) {
                        avg_array[j] += scores[idx] * arr[j];
                    }
                }
                for (double& v : avg_array) {
                    v /= vote_score_sum;
                }
                BBox bbox = BBox::from_vector({avg_array[0], avg_array[1], avg_array[2],
                                               avg_array[3], avg_array[4], avg_array[5],
                                               avg_array[6], scores[index]});
                bbox.s = scores[index];
                result_indexes.push_back(index);
                result_types.push_back(inst_types[index]);
            } else {
                result_indexes.push_back(index);
                result_types.push_back(inst_types[index]);
            }
        } else {
            result_indexes.push_back(index);
            result_types.push_back(inst_types[index]);
        }

        std::vector<int> new_order;
        for (int idx : order) {
            if (std::find(related_inds.begin(), related_inds.end(), idx) == related_inds.end()) {
                new_order.push_back(idx);
            }
        }
        order.swap(new_order);
    }

    return {result_indexes, result_types};
}

}  // namespace simpletrack
