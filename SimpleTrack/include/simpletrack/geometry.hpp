// Geometry utilities translated from mot_3d/utils/geometry.py
#pragma once

#include <Eigen/Dense>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/multi_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "simpletrack/bbox.hpp"

namespace simpletrack {

namespace bg = boost::geometry;

using Point2D = bg::model::d2::point_xy<double>;
using Polygon2D = bg::model::polygon<Point2D, false, true>;  // counter-clockwise, closed

Polygon2D bbox_to_polygon(const BBox& box);

double polygon_area(const Polygon2D& poly);

double diff_orientation_correction(double diff);

double m_distance(const BBox& det,
                  const BBox& trk,
                  const std::optional<Eigen::Matrix<double, 7, 7>>& inv_innovation = std::nullopt);

std::pair<double, double> iou3d(const BBox& box_a, const BBox& box_b);

double giou3d(const BBox& box_a, const BBox& box_b);

}  // namespace simpletrack
