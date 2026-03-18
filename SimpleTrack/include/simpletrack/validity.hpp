// Translated from mot_3d/data_protos/validity.py
#pragma once

#include <string>

namespace simpletrack {

class Validity {
public:
    static bool valid(const std::string& state_string);
    static bool not_output(const std::string& state_string);
    static bool predicted(const std::string& state_string);
    static std::string modify_string(const std::string& state_string, int mode);
};

}  // namespace simpletrack

