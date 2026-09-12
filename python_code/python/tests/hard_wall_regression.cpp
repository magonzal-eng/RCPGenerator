#include "rcpgenerator/initialize_particles.hpp"
#include "rcpgenerator/rcp_generator.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_finite(double value, const char* label) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(label) + " is not finite");
    }
}

template <typename Matrix>
void require_all_finite(const Matrix& values, const char* label) {
    for (const auto& row : values) {
        for (double value : row) {
            require_finite(value, label);
        }
    }
}

void require_all_finite(const std::vector<double>& values, const char* label) {
    for (double value : values) {
        require_finite(value, label);
    }
}

void require_true(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, double tolerance,
                   const char* label) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            std::string(label) + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
    }
}

std::size_t candidate_pair_count(const std::vector<std::int8_t>& walls) {
    constexpr std::size_t N = 2;
    constexpr std::size_t Ndim = 2;
    const std::vector<double> positions = {0.15, 0.50, 0.85, 0.50};
    const std::vector<double> diameters = {0.30, 0.30};
    const std::vector<double> box = {1.0, 1.0};
    const std::vector<std::uint32_t> refresh = {1, 1};
    const std::vector<std::uint32_t> capacities = {4, 4};
    const std::vector<std::uint64_t> offsets = {0, 5, 10};
    std::vector<std::uint32_t> pairs(offsets.back(), 0);

    rcpgenerator::get_pairs_nd_3(
        N, Ndim, positions.data(), diameters, box, walls, refresh,
        capacities.data(), offsets.data(), pairs.data(), nullptr, nullptr);

    return static_cast<std::size_t>(pairs[offsets[0]]) +
           static_cast<std::size_t>(pairs[offsets[1]]);
}

void test_candidate_geometry_respects_walls() {
    require_true(candidate_pair_count({0, 0}) > 0,
                 "periodic x-axis did not create the expected image candidate");
    require_true(candidate_pair_count({1, 0}) == 0,
                 "hard x-axis incorrectly created a periodic image candidate");
}

void test_force_geometry_respects_walls() {
    constexpr std::size_t N = 2;
    constexpr std::size_t Ndim = 2;
    const std::vector<double> positions = {0.10, 0.50, 0.90, 0.50};
    const std::vector<double> diameters = {0.30, 0.30};
    const std::vector<double> box = {1.0, 1.0};
    const std::vector<std::int8_t> walls = {1, 0};

    // Store one deliberately supplied cross-domain pair. With a hard x-axis,
    // it must not create a particle-particle force. The only expected forces
    // are the two equal wall-restoring forces caused by the 0.05 penetration.
    const std::vector<std::uint64_t> offsets = {0, 2, 3};
    const std::vector<std::uint32_t> pairs = {1, 1, 0};
    std::vector<double> forces(N * Ndim, 0.0);
    std::vector<std::vector<double>> min_dist;
    std::vector<std::size_t> contacts(N, 0);
    double energy = 0.0;
    double max_overlap = 0.0;
    double contact_length = 0.0;
    double mean_force = 0.0;
    double diameter_gradient = 0.0;

    rcpgenerator::get_forces_nd_3(
        pairs.data(), offsets.data(), positions.data(), N, Ndim, diameters,
        box, walls, forces.data(), energy, min_dist, max_overlap,
        contact_length, mean_force, 0.0, diameter_gradient, contacts);

    require_close(forces[0], 0.10, 1e-12, "left hard-wall force");
    require_close(forces[1], 0.00, 1e-12, "left transverse force");
    require_close(forces[2], -0.10, 1e-12, "right hard-wall force");
    require_close(forces[3], 0.00, 1e-12, "right transverse force");
}

}  // namespace

int main() {
    try {
        test_candidate_geometry_respects_walls();
        test_force_geometry_respects_walls();

        rcpgenerator::InitializerConfig init_config;
        init_config.phi = 0.11;
        init_config.N = 64;
        init_config.Ndim = 2;
        init_config.box = {1.0, 1.0};
        init_config.walls = {1, 1};
        init_config.fix_height = false;
        init_config.dist.type = "mono";
        init_config.dist.d = 1.0;

        auto initialized = rcpgenerator::initialize_particles(init_config);
        require_all_finite(initialized.positions, "initialized position");
        require_all_finite(initialized.diameters, "initialized diameter");
        require_all_finite(initialized.box, "initialized box");
        require_finite(initialized.diameter_scale_factor, "diameter scale factor");
        require_finite(initialized.phi_modifier, "phi modifier");

        rcpgenerator::PackingInput input;
        input.positions = initialized.positions;
        input.diameters = initialized.diameters;

        rcpgenerator::PackingConfig pack_config;
        pack_config.box = initialized.box;
        pack_config.walls = initialized.walls;
        pack_config.fix_height = false;
        pack_config.neighbor_max = 0;
        pack_config.seed = 123u;

        rcpgenerator::PackingRunOptions options;
        options.max_steps = 500;

        auto observed = rcpgenerator::run_packing_observed(
            input,
            pack_config,
            0,
            false,
            0,
            rcpgenerator::PackingObserver(),
            options);

        const auto& result = observed.first;
        require_all_finite(result.positions, "packed position");
        require_all_finite(result.diameters, "packed diameter");
        require_all_finite(result.box, "packed box");
        require_finite(result.phi, "packed phi");
        require_finite(result.max_min_dist, "packed max_min_dist");
        require_finite(result.force_magnitude, "packed force_magnitude");

        std::cout << "Hard-wall regression checks passed.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Hard-wall regression failure: " << ex.what() << '\n';
        return 1;
    }
}
