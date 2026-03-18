// Copyright (c) 2024 SimpleTrack
// C++ translation of mot_3d/data_protos/bbox.py utilities
#pragma once

#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace simpletrack {

struct BBox {
    double x{0.0};  // center x (meters)
    double y{0.0};  // center y (meters)
    double z{0.0};  // center z (meters)
    double h{0.0};  // height  (meters)
    double w{0.0};  // width   (meters)
    double l{0.0};  // length  (meters)
    double o{0.0};  // yaw / heading (radians)
    double s{std::numeric_limits<double>::quiet_NaN()};  // detection score (optional)

    BBox() = default;

    BBox(double x_, double y_, double z_, double h_, double w_, double l_, double o_, double s_ = std::numeric_limits<double>::quiet_NaN())
        : x(x_), y(y_), z(z_), h(h_), w(w_), l(l_), o(o_), s(s_) {}

    bool has_score() const noexcept {
        return !std::isnan(s);
    }

    std::array<double, 8> to_array_with_score() const noexcept {
        return {x, y, z, o, l, w, h, has_score() ? s : 0.0};
    }

    std::array<double, 7> to_array() const noexcept {
        return {x, y, z, o, l, w, h};
    }

    static BBox from_array(const std::initializer_list<double>& data) {
        if (data.size() < 7) {
            throw std::invalid_argument("BBox::from_array expects at least 7 elements");
        }
        auto it = data.begin();
        BBox box;
        box.x = *it++; box.y = *it++; box.z = *it++; box.o = *it++;
        box.l = *it++; box.w = *it++; box.h = *it++;
        if (it != data.end()) {
            box.s = *it;
        }
        return box;
    }

    static BBox from_vector(const std::vector<double>& data) {
        if (data.size() < 7) {
            throw std::invalid_argument("BBox::from_vector expects at least 7 elements");
        }
        BBox box;
        box.x = data[0];
        box.y = data[1];
        box.z = data[2];
        box.o = data[3];
        box.l = data[4];
        box.w = data[5];
        box.h = data[6];
        if (data.size() > 7) {
            box.s = data[7];
        }
        return box;
    }
};

inline std::vector<std::array<double, 3>> box2corners2d(const BBox& bbox) {
    const double cos_yaw = std::cos(bbox.o);
    const double sin_yaw = std::sin(bbox.o);
    const double half_l = bbox.l * 0.5;
    const double half_w = bbox.w * 0.5;
    const double z_bottom = bbox.z - bbox.h * 0.5;

    auto corner = [&](double l_off, double w_off) {
        return std::array<double, 3>{
            bbox.x + cos_yaw * l_off + sin_yaw * w_off,
            bbox.y + sin_yaw * l_off - cos_yaw * w_off,
            z_bottom
        };
    };

    std::vector<std::array<double, 3>> corners;
    corners.reserve(4);
    corners.push_back(corner(half_l, half_w));
    corners.push_back(corner(half_l, -half_w));
    corners.push_back(corner(-half_l, -half_w));
    corners.push_back(corner(-half_l, half_w));
    return corners;
}

}  // namespace simpletrack
