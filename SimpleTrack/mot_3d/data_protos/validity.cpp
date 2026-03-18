#include "simpletrack/validity.hpp"

#include <sstream>
#include <stdexcept>
#include <vector>

namespace simpletrack {

namespace {
std::vector<std::string> split_tokens(const std::string& input, char delimiter = '_') {
    std::vector<std::string> tokens;
    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        if (!item.empty()) {
            tokens.push_back(item);
        }
    }
    return tokens;
}
}  // namespace

bool Validity::valid(const std::string& state_string) {
    auto tokens = split_tokens(state_string);
    if (tokens.empty()) {
        return false;
    }
    if (tokens[0] == "birth") {
        return true;
    }
    if (tokens.size() < 3) {
        return false;
    }
    if (tokens[0] == "alive") {
        try {
            return std::stoi(tokens[1]) == 1;
        } catch (...) {
            return false;
        }
    }
    return false;
}

bool Validity::not_output(const std::string& state_string) {
    auto tokens = split_tokens(state_string);
    if (tokens.size() < 3) {
        return false;
    }
    if (tokens[0] == "alive") {
        try {
            return std::stoi(tokens[1]) != 1;
        } catch (...) {
            return false;
        }
    }
    return false;
}

bool Validity::predicted(const std::string& state_string) {
    auto tokens = split_tokens(state_string);
    if (tokens.size() < 2) {
        throw std::invalid_argument("state_string is malformed");
    }
    const std::string& state = tokens[0];
    if (state != "birth" && state != "alive" && state != "death" && state != "dead") {
        throw std::invalid_argument("type name not existed");
    }
    if (state == "alive") {
        int token_value = std::stoi(tokens[1]);
        return token_value != 0;
    }
    return false;
}

std::string Validity::modify_string(const std::string& state_string, int mode) {
    auto tokens = split_tokens(state_string);
    if (tokens.size() < 3) {
        throw std::invalid_argument("state_string must contain at least three tokens");
    }
    tokens[1] = std::to_string(mode);
    return tokens[0] + "_" + tokens[1] + "_" + tokens[2];
}

}  // namespace simpletrack
