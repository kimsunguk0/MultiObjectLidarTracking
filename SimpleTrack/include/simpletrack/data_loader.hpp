#pragma once

#include <optional>
#include <string>
#include <vector>

#include "simpletrack/frame_data.hpp"

namespace simpletrack {

class SequenceLoader {
public:
    SequenceLoader() = default;
    explicit SequenceLoader(std::vector<FrameData> frames)
        : frames_(std::move(frames)) {}

    void reset() { index_ = 0; }
    bool has_next() const { return index_ < frames_.size(); }
    std::optional<FrameData> next();
    size_t size() const { return frames_.size(); }

protected:
    std::vector<FrameData> frames_;
    size_t index_{0};
};

class WaymoLoader : public SequenceLoader {
public:
    using SequenceLoader::SequenceLoader;
};

class NuScenesLoader : public SequenceLoader {
public:
    using SequenceLoader::SequenceLoader;
};

}  // namespace simpletrack

