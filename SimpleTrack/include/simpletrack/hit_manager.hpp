// HitManager finite state machine translated from mot_3d/life/hit_manager.py
#pragma once

#include <string>

#include "simpletrack/config.hpp"
#include "simpletrack/update_info_data.hpp"
#include "simpletrack/validity.hpp"

namespace simpletrack {

class HitManager {
public:
    HitManager(const TrackerConfig& configs, int frame_index);

    void predict(bool is_key_frame);
    int if_valid(const UpdateInfoData& update_info);
    void update(const UpdateInfoData& update_info, bool is_key_frame = true);
    bool alive(int frame_index) const;
    bool death(int frame_index) const;
    bool valid_output(int frame_index) const;
    std::string state_string(int frame_index) const;

    int time_since_update{0};
    int hits{1};
    int hit_streak{1};
    int first_continuing_hit{1};
    bool still_first{true};
    int age{0};
    int recent_state{1};
    bool no_asso{false};
    bool fall{false};
    std::string state{"birth"};

private:
    void state_transition(int mode, int frame_index);

    int max_age_{3};
    int min_hits_{3};
};

}  // namespace simpletrack

