#pragma once

#include <cmath>

namespace simpletrack {

constexpr int kBEVGridSize = 1216;

struct Boundary {
    double minX{-25.0};
    double maxX{75.0};
    double minY{-50.0};
    double maxY{50.0};
    double minZ{-4.1};
    double maxZ{1.4};

    double discretization() const {
        return (maxX - minX) / static_cast<double>(kBEVGridSize);
    }

    double z_range() const {
        return std::abs(maxZ - minZ);
    }
};

inline const Boundary kDefaultBoundary{};

}  // namespace simpletrack

