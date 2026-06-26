/*
 * sdf_module.cpp
 */
// Source - https://stackoverflow.com/a/2582597
// Posted by Brian R. Bondy, modified by community. See post 'Timeline' for change history
// Retrieved 2026-06-16, License - CC BY-SA 2.5

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <vector>
#include <stdexcept>
#include <functional>
#include <iostream>

namespace py = pybind11;

static py::object       g_bullet     = py::none();
static py::object       g_arm_id     = py::none();
static std::vector<int> g_body_ids;
static int              g_num_joints = 0;
static double           g_query_dist = 1.5;
static bool             g_alive      = false;

static void check_initialised() {
    if (!g_alive || g_bullet.is_none())
        throw std::runtime_error("sdf_module: call init() before querying the SDF.");
}

// Query minimum signed distance from arm to any reconstructed body.
// Uses p.getClosestPoints exactly as in your pipeline.
// GIL must be held by caller (it always is — we never release it).
static double _query(const double* config) {
    if (!g_alive) return g_query_dist;

    for (int j = 0; j < g_num_joints; ++j)
        g_bullet.attr("resetJointState")(g_arm_id, j, config[j]);
    g_bullet.attr("stepSimulation")();

    double min_d = g_query_dist;

    for (int body_id : g_body_ids) {
        py::object contacts = g_bullet.attr("getClosestPoints")(
            py::arg("bodyA")    = g_arm_id,
            py::arg("bodyB")    = body_id,
            py::arg("distance") = g_query_dist
        );
        if (!contacts.is_none()) {
            for (auto contact : contacts) {
                double d = contact.attr("__getitem__")(8).cast<double>();
                if (d < min_d) {
                    min_d = d;
                    if (min_d < 0.0) return min_d;
                }
            }
        }
    }
    return min_d;
}

static void init(
    py::object       bullet_module,
    int              arm_id,
    std::vector<int> body_ids,
    int              num_joints,
    double           query_dist = 1.5
) {
    g_bullet     = bullet_module;
    g_arm_id     = py::int_(arm_id);
    g_body_ids   = std::move(body_ids);
    g_num_joints = num_joints;
    g_query_dist = query_dist;
    g_alive      = true;
    std::cout << "[sdf_module] Initialised: arm_id=" << arm_id
              << "  bodies=" << g_body_ids.size()
              << "  joints=" << num_joints
              << "  query_dist=" << query_dist << "m\n";
}

// Call before p.disconnect() to prevent GIL crash at interpreter shutdown.
static void shutdown() {
    g_alive  = false;
    g_bullet = py::none();
    g_arm_id = py::none();
    g_body_ids.clear();
    std::cout << "[sdf_module] Shutdown.\n";
}

static double query_sdf(const std::vector<double>& config) {
    check_initialised();
    if ((int)config.size() != g_num_joints)
        throw std::invalid_argument("query_sdf: config length != num_joints");
    return _query(config.data());
}

static std::function<double(const double*)> get_sdf_fn() {
    check_initialised();
    return [](const double* cfg) -> double { return _query(cfg); };
}

PYBIND11_MODULE(sdf_module, m) {
    m.doc() = "PyBullet SDF bridge for manifold tracing.";
    m.def("init", &init,
          py::arg("bullet_module"), py::arg("arm_id"), py::arg("body_ids"),
          py::arg("num_joints"), py::arg("query_dist") = 1.5);
    m.def("shutdown", &shutdown,
          "Call before p.disconnect() to avoid GIL crash at exit.");
    m.def("query_sdf", &query_sdf, py::arg("config"));
    m.def("get_sdf_fn", &get_sdf_fn);
}