#include "simpletrack/visualizer2d.hpp"

#include <array>
#include <fstream>
#include <iostream>

namespace simpletrack {

Visualizer2D::Visualizer2D(std::string name, std::pair<int, int>)
    : name_(std::move(name)) {}

void Visualizer2D::show() {
    std::cout << "[Visualizer2D] show() requested for '" << name_
              << "' but GUI rendering is not available in the C++ port.\n";
}

void Visualizer2D::close() {
    point_cloud_.clear();
    boxes_.clear();
}

bool Visualizer2D::save(const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "[Visualizer2D] failed to open " << path << " for writing.\n";
        return false;
    }
    ofs << "# Visualizer2D dump for '" << name_ << "'\n";
    ofs << "# point_cloud size=" << point_cloud_.size() << "\n";
    for (const auto& p : point_cloud_) {
        ofs << "PC " << p[0] << ' ' << p[1] << ' ' << p[2] << "\n";
    }
    ofs << "# boxes=" << boxes_.size() << "\n";
    for (const auto& entry : boxes_) {
        ofs << "BOX " << entry.box.x << ' ' << entry.box.y << ' ' << entry.box.z
            << ' ' << entry.box.l << ' ' << entry.box.w << ' ' << entry.box.h
            << ' ' << entry.box.o << ' ' << entry.box.s
            << " msg='" << entry.message << "' color=" << entry.color
            << " linestyle=" << entry.linestyle << "\n";
    }
    return true;
}

void Visualizer2D::handler_pc(const std::vector<std::array<double, 3>>& pc, const std::string&) {
    point_cloud_.insert(point_cloud_.end(), pc.begin(), pc.end());
}

void Visualizer2D::handler_box(const BBox& box, const std::string& message,
                               const std::string& color, const std::string& linestyle) {
    boxes_.push_back(BoxEntry{box, message, color, linestyle});
}

}  // namespace simpletrack
