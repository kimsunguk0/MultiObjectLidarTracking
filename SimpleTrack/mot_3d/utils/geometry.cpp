// Geometry utilities translated from mot_3d/utils/geometry.py

#include "simpletrack/geometry.hpp"

#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/convex_hull.hpp>
#include <boost/geometry/algorithms/intersection.hpp>
#include <boost/geometry/algorithms/within.hpp>
#include <boost/geometry/geometries/segment.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace simpletrack {

namespace bg = boost::geometry;

Polygon2D bbox_to_polygon(const BBox& box) {
    Polygon2D poly;
    auto corners = box2corners2d(box);
    auto& outer = poly.outer();
    outer.reserve(5);
    for (const auto& c : corners) {
        outer.emplace_back(c[0], c[1]);
    }
    // close polygon
    outer.emplace_back(corners.front()[0], corners.front()[1]);
    bg::correct(poly);
    return poly;
}

double polygon_area(const Polygon2D& poly) {
    return std::abs(bg::area(poly));
}

double diff_orientation_correction(double diff) {
    constexpr double kHalfPi = M_PI / 2.0;
    if (diff > kHalfPi) {
        diff -= M_PI;
    }
    if (diff < -kHalfPi) {
        diff += M_PI;
    }
    return diff;
}

double m_distance(const BBox& det,
                  const BBox& trk,
                  const std::optional<Eigen::Matrix<double, 7, 7>>& inv_innovation) {
    Eigen::Matrix<double, 7, 1> det_vec;
    Eigen::Matrix<double, 7, 1> trk_vec;

    auto det_arr = det.to_array();
    auto trk_arr = trk.to_array();
    for (int i = 0; i < 7; ++i) {
        det_vec(i) = det_arr[i];
        trk_vec(i) = trk_arr[i];
    }

    Eigen::Matrix<double, 7, 1> diff = det_vec - trk_vec;
    diff(3) = diff_orientation_correction(diff(3));

    if (inv_innovation) {
        const auto& inv = *inv_innovation;
        double value = (diff.transpose() * inv * diff)(0, 0);
        return std::sqrt(std::max(0.0, value));
    }
    return diff.norm();
}

std::pair<double, double> iou3d(const BBox& box_a, const BBox& box_b) {
    Polygon2D poly_a = bbox_to_polygon(box_a);
    Polygon2D poly_b = bbox_to_polygon(box_b);

    std::vector<Polygon2D> intersection;
    bg::intersection(poly_a, poly_b, intersection);

    double overlap_area = 0.0;
    for (const auto& poly : intersection) {
        overlap_area += polygon_area(poly);
    }

    double area_a = polygon_area(poly_a);
    double area_b = polygon_area(poly_b);
    double union_area = area_a + area_b - overlap_area;
    double iou2d = union_area > 0.0 ? overlap_area / union_area : 0.0;

    const double ha = box_a.h;
    const double hb = box_b.h;
    const double za_min = box_a.z - ha * 0.5;
    const double za_max = box_a.z + ha * 0.5;
    const double zb_min = box_b.z - hb * 0.5;
    const double zb_max = box_b.z + hb * 0.5;

    const double overlap_height = std::max(0.0, std::min(za_max, zb_max) - std::max(za_min, zb_min));
    const double overlap_volume = overlap_area * overlap_height;
    const double volume_a = box_a.w * box_a.l * ha;
    const double volume_b = box_b.w * box_b.l * hb;
    const double union_volume = volume_a + volume_b - overlap_volume;
    double iou3d_val = union_volume > 0.0 ? overlap_volume / union_volume : 0.0;

    return {iou2d, iou3d_val};
}

double giou3d(const BBox& box_a, const BBox& box_b) {
    Polygon2D poly_a = bbox_to_polygon(box_a);
    Polygon2D poly_b = bbox_to_polygon(box_b);

    std::vector<Polygon2D> intersection;
    bg::intersection(poly_a, poly_b, intersection);

    double intersection_area = 0.0;
    for (const auto& poly : intersection) {
        intersection_area += polygon_area(poly);
    }

    double area_a = polygon_area(poly_a);
    double area_b = polygon_area(poly_b);
    double union_area = area_a + area_b - intersection_area;

    // Compute convex hull area
    bg::model::multi_point<Point2D> combined;
    combined.reserve(poly_a.outer().size() + poly_b.outer().size());
    for (const auto& p : poly_a.outer()) {
        combined.emplace_back(bg::get<0>(p), bg::get<1>(p));
    }
    for (const auto& p : poly_b.outer()) {
        combined.emplace_back(bg::get<0>(p), bg::get<1>(p));
    }

    Polygon2D hull;
    bg::convex_hull(combined, hull);
    double convex_area = polygon_area(hull);

    const double ha = box_a.h;
    const double hb = box_b.h;
    const double za_min = box_a.z - ha * 0.5;
    const double za_max = box_a.z + ha * 0.5;
    const double zb_min = box_b.z - hb * 0.5;
    const double zb_max = box_b.z + hb * 0.5;

    const double overlap_height_candidate1 = (za_max) - (zb_min);
    const double overlap_height_candidate2 = (zb_max) - (za_min);
    const double overlap_height = std::max(0.0, std::min({overlap_height_candidate1, overlap_height_candidate2, ha, hb}));
    const double union_height = std::max({overlap_height_candidate1, overlap_height_candidate2, ha, hb, 1e-9});

    const double intersection_volume = intersection_area * overlap_height;
    const double volume_a = box_a.w * box_a.l * ha;
    const double volume_b = box_b.w * box_b.l * hb;
    const double union_volume = volume_a + volume_b - intersection_volume;

    const double convex_volume = convex_area * union_height;
    if (convex_volume <= 0.0) {
        return 0.0;
    }

    const double iou = union_volume > 0.0 ? intersection_volume / union_volume : 0.0;
    const double giou = iou - (convex_volume - union_volume) / convex_volume;
    return giou;
}

}  // namespace simpletrack
