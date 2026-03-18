#pragma once

#include <array>
#include <string>
#include <vector>

#include "simpletrack/bbox.hpp"

namespace simpletrack {

class Visualizer2D {
public:
    Visualizer2D(std::string name = "", std::pair<int, int> /*figsize*/ = {8, 8});

    void show();
    void close();
    bool save(const std::string& path);

    void handler_pc(const std::vector<std::array<double, 3>>& pc, const std::string& color = "gray");
    void handler_box(const BBox& box, const std::string& message = "", const std::string& color = "red", const std::string& linestyle = "solid");

private:
    std::string name_;
    std::vector<std::array<double, 3>> point_cloud_;
    struct BoxEntry {
        BBox box;
        std::string message;
        std::string color;
        std::string linestyle;
    };
    std::vector<BoxEntry> boxes_;
};

}  // namespace simpletrack
