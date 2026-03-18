#include "simpletrack/hit_manager.hpp"

#include <algorithm>

namespace simpletrack {

HitManager::HitManager(const TrackerConfig& configs, int frame_index) {
    if (configs.running.max_age_since_update > 0) {
        max_age_ = configs.running.max_age_since_update;
    }
    if (configs.running.min_hits_to_birth > 0) {
        min_hits_ = configs.running.min_hits_to_birth;
    }

    if (frame_index <= min_hits_ || min_hits_ == 0) {
        state = "alive";
        recent_state = 1;
    }
}

void HitManager::predict(bool is_key_frame) {
    if (!is_key_frame) {
        return;
    }
    age += 1;
    if (time_since_update > 0) {
        hit_streak = 0;
        still_first = false;
    }
    time_since_update += 1;
    fall = true;
}

int HitManager::if_valid(const UpdateInfoData& update_info) {
    recent_state = update_info.mode;
    return update_info.mode;
}

void HitManager::update(const UpdateInfoData& update_info, bool is_key_frame) {
    int association = if_valid(update_info);
    recent_state = association;
    if (association != 0) {
        fall = false;
        time_since_update = 0;
        hits += 1;
        hit_streak += 1;
        if (still_first) {
            first_continuing_hit += 1;
        }
    }
    if (is_key_frame) {
        state_transition(association, update_info.frame_index);
    }
}

void HitManager::state_transition(int /*mode*/, int frame_index) {
    if (state == "birth") {
        if ((hits >= min_hits_) || (frame_index <= min_hits_)) {
            state = "alive";
            // recent_state already updated
        } else if (time_since_update >= max_age_) {
            state = "dead";
        }
    } else if (state == "alive") {
        if (time_since_update >= max_age_) {
            state = "dead";
        }
    }
}

bool HitManager::alive(int /*frame_index*/) const {
    return state == "alive";
}

bool HitManager::death(int /*frame_index*/) const {
    return state == "dead";
}

bool HitManager::valid_output(int /*frame_index*/) const {
    return (state == "alive") && (!no_asso);
}

std::string HitManager::state_string(int /*frame_index*/) const {
    if (state == "birth") {
        return state + "_" + std::to_string(hits);
    }
    if (state == "alive") {
        return state + "_" + std::to_string(recent_state) + "_" + std::to_string(time_since_update);
    }
    return state + "_" + std::to_string(time_since_update);
}

}  // namespace simpletrack
