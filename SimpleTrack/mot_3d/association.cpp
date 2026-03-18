#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <omp.h>

#include "simpletrack/association.hpp"
#include "simpletrack/bbox.hpp"
#include "simpletrack/geometry.hpp"

namespace simpletrack {

using Matrix = Eigen::MatrixXd;
using MatchList = std::vector<std::pair<int, int>>;

Matrix compute_iou_distance(const std::vector<BBox>& dets,
                            const std::vector<BBox>& tracks,
                            const std::string& asso) {
    const int num_dets = static_cast<int>(dets.size());
    const int num_trks = static_cast<int>(tracks.size());
    Matrix dist = Matrix::Zero(num_dets, num_trks);
#pragma omp parallel for collapse(2)
    for (int d = 0; d < num_dets; ++d) {
        for (int t = 0; t < num_trks; ++t) {
            if (asso == "iou") {
                dist(d, t) = 1.0 - iou3d(dets[d], tracks[t]).second;
            } else if (asso == "giou") {
                const double giou = giou3d(dets[d], tracks[t]);
                dist(d, t) = 1.0 - giou;
            } else {
                // collapse pragma requires all branches; guard to avoid throwing concurrently
                dist(d, t) = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    if (asso != "iou" && asso != "giou") {
        throw std::invalid_argument("Unsupported association metric: " + asso);
    }
    return dist;
}

Matrix compute_m_distance(const std::vector<BBox>& dets,
                          const std::vector<BBox>& tracks,
                          const std::vector<Eigen::Matrix<double, 7, 7>>* trk_innovation) {
    const int num_dets = static_cast<int>(dets.size());
    const int num_trks = static_cast<int>(tracks.size());
    Matrix dist = Matrix::Zero(num_dets, num_trks);
    for (int i = 0; i < num_dets; ++i) {
        for (int j = 0; j < num_trks; ++j) {
            if (trk_innovation) {
                dist(i, j) = m_distance(dets[i], tracks[j], (*trk_innovation)[j]);
            } else {
                dist(i, j) = m_distance(dets[i], tracks[j], std::nullopt);
            }
        }
    }
    return dist;
}

namespace {

struct MatcherOutput {
    MatchList matches;
    Matrix distance_matrix;
};

class HungarianSolver {
public:
    explicit HungarianSolver(const Matrix& cost_matrix) {
        const int rows = static_cast<int>(cost_matrix.rows());
        const int cols = static_cast<int>(cost_matrix.cols());
        size_ = std::max(rows, cols);
        cost_ = Matrix::Constant(size_, size_, cost_matrix.maxCoeff());
        cost_.block(0, 0, rows, cols) = cost_matrix;
        original_rows_ = rows;
        original_cols_ = cols;
    }

    MatchList solve() {
        const int n = size_;
        const double INF = std::numeric_limits<double>::infinity();
        std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
        std::vector<int> p(n + 1, 0), way(n + 1, 0);

        for (int i = 1; i <= n; ++i) {
            p[0] = i;
            int j0 = 0;
            std::vector<double> minv(n + 1, INF);
            std::vector<bool> used(n + 1, false);
            do {
                used[j0] = true;
                int i0 = p[j0];
                int j1 = 0;
                double delta = INF;
                for (int j = 1; j <= n; ++j) {
                    if (used[j]) continue;
                    double cur = cost_(i0 - 1, j - 1) - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
                for (int j = 0; j <= n; ++j) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
            } while (p[j0] != 0);

            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }

        MatchList assignment;
        assignment.reserve(size_);
        int rows = original_rows_;
        int cols = original_cols_;
        for (int j = 1; j <= size_; ++j) {
            if (p[j] <= rows && j <= cols) {
                assignment.emplace_back(p[j] - 1, j - 1);
            }
        }
        return assignment;
    }

private:
    Matrix cost_;
    int size_{0};
    int original_rows_{0};
    int original_cols_{0};
};

MatcherOutput bipartite_matcher(const std::vector<BBox>& dets,
                                const std::vector<BBox>& tracks,
                                const std::string& asso,
                                double /*dist_threshold*/,
                                const std::vector<Eigen::Matrix<double, 7, 7>>* trk_innovation) {
    Matrix dist_matrix;
    if (asso == "iou" || asso == "giou") {
        dist_matrix = compute_iou_distance(dets, tracks, asso);
    } else if (asso == "m_dis") {
        dist_matrix = compute_m_distance(dets, tracks, trk_innovation);
    } else if (asso == "euler") {
        dist_matrix = compute_m_distance(dets, tracks, nullptr);
    } else {
        throw std::invalid_argument("Unsupported association metric: " + asso);
    }
    if (dets.empty() || tracks.empty()) {
        return {MatchList{}, dist_matrix};
    }
    HungarianSolver solver(dist_matrix);
    MatchList matches = solver.solve();
    return {matches, dist_matrix};
}

MatcherOutput greedy_matcher(const std::vector<BBox>& dets,
                             const std::vector<BBox>& tracks,
                             const std::string& asso,
                             double /*dist_threshold*/,
                             const std::vector<Eigen::Matrix<double, 7, 7>>* trk_innovation) {
    Matrix distance_matrix;
    if (asso == "m_dis") {
        distance_matrix = compute_m_distance(dets, tracks, trk_innovation);
    } else if (asso == "euler") {
        distance_matrix = compute_m_distance(dets, tracks, nullptr);
    } else if (asso == "iou" || asso == "giou") {
        distance_matrix = compute_iou_distance(dets, tracks, asso);
    } else {
        throw std::invalid_argument("Unsupported association metric: " + asso);
    }
    if (dets.empty() || tracks.empty()) {
        return {MatchList{}, distance_matrix};
    }

    const int num_dets = static_cast<int>(dets.size());
    const int num_trks = static_cast<int>(tracks.size());
    std::vector<int> det_assignment(num_dets, -1);
    std::vector<int> trk_assignment(num_trks, -1);

    struct Entry {
        double distance;
        int det_idx;
        int trk_idx;
    };

    std::vector<Entry> entries;
    entries.reserve(num_dets * num_trks);
    for (int i = 0; i < num_dets; ++i) {
        for (int j = 0; j < num_trks; ++j) {
            entries.push_back({distance_matrix(i, j), i, j});
        }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.distance < b.distance;
    });

    MatchList matches;
    for (const auto& e : entries) {
        if (det_assignment[e.det_idx] == -1 && trk_assignment[e.trk_idx] == -1) {
            matches.emplace_back(e.det_idx, e.trk_idx);
            det_assignment[e.det_idx] = e.trk_idx;
            trk_assignment[e.trk_idx] = e.det_idx;
        }
    }

    return {matches, distance_matrix};
}

}  // namespace

AssociationResult associate_dets_to_tracks(
    const std::vector<BBox>& dets,
    const std::vector<BBox>& tracks,
    const std::string& mode,
    const std::string& asso,
    double dist_threshold,
    const std::vector<Eigen::Matrix<double, 7, 7>>* trk_innovation_matrix) {

    MatcherOutput output;
    if (mode == "bipartite") {
        output = bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix);
    } else if (mode == "greedy") {
        output = greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix);
    } else {
        throw std::invalid_argument("Unsupported association mode: " + mode);
    }

    const int num_dets = static_cast<int>(dets.size());
    const int num_trks = static_cast<int>(tracks.size());

    std::vector<int> unmatched_dets;
    std::vector<int> unmatched_tracks;
    unmatched_dets.reserve(num_dets);
    unmatched_tracks.reserve(num_trks);

    std::vector<bool> det_matched(num_dets, false);
    std::vector<bool> trk_matched(num_trks, false);

    MatchList filtered_matches;
    filtered_matches.reserve(output.matches.size());

    for (const auto& match : output.matches) {
        const int det_idx = match.first;
        const int trk_idx = match.second;
        if (det_idx < 0 || det_idx >= num_dets || trk_idx < 0 || trk_idx >= num_trks) {
            continue;
        }
        const double distance = output.distance_matrix(det_idx, trk_idx);
        if (distance > dist_threshold) {
            continue;
        }
        filtered_matches.emplace_back(det_idx, trk_idx);
        det_matched[det_idx] = true;
        trk_matched[trk_idx] = true;
    }

    for (int i = 0; i < num_dets; ++i) {
        if (!det_matched[i]) {
            unmatched_dets.push_back(i);
        }
    }
    for (int j = 0; j < num_trks; ++j) {
        if (!trk_matched[j]) {
            unmatched_tracks.push_back(j);
        }
    }

    return {filtered_matches, unmatched_dets, unmatched_tracks};
}

}  // namespace simpletrack
