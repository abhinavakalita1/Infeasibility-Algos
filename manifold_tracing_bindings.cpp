#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "utils.h"
#include "permutahedral_simplex.h"
#include "fk_triangulation.h"
#include "manifold_tracing.h"   // the header containing edge_intersection, traceManifold, etc.

#include <unordered_map>
#include <vector>
#include <array>

namespace py = pybind11;

using Point = std::array<double, MAX_D>;

// ---------------------------------------------------------------------------
// Helpers: convert numpy arrays <-> raw C arrays
// ---------------------------------------------------------------------------

static void np_to_double(py::array_t<double> arr, double* out, int n) {
    auto r = arr.unchecked<1>();
    for (int i = 0; i < n; ++i) out[i] = r(i);
}

static py::array_t<double> double_to_np(const double* arr, int n) {
    py::array_t<double> result(n);
    auto r = result.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) r(i) = arr[i];
    return result;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(manifold_tracing, m) {
    m.doc() = "Python bindings for Manifold Tracing via Permutahedral FK Triangulation";

    // -----------------------------------------------------------------------
    // Permutahedral_Simplex
    // -----------------------------------------------------------------------
    py::class_<Permutahedral_Simplex>(m, "PermutahedralSimplex")
        .def(py::init<>())
        // anchor as a list of ints
        .def_property("anchor",
            [](const Permutahedral_Simplex& s) {
                std::vector<int32_t> v(s.anchor, s.anchor + s.amb_dim);
                return v;
            },
            [](Permutahedral_Simplex& s, std::vector<int32_t> v) {
                for (int i = 0; i < (int)v.size() && i < MAX_D; ++i)
                    s.anchor[i] = v[i];
            })
        .def_readwrite("amb_dim",    &Permutahedral_Simplex::amb_dim)
        .def_readwrite("num_blocks", &Permutahedral_Simplex::num_blocks)
        // block_sizes as a list
        .def_property("block_sizes",
            [](const Permutahedral_Simplex& s) {
                std::vector<uint8_t> v(s.block_sizes, s.block_sizes + s.num_blocks);
                return v;
            },
            [](Permutahedral_Simplex& s, std::vector<uint8_t> v) {
                for (int i = 0; i < (int)v.size() && i <= MAX_D; ++i)
                    s.block_sizes[i] = v[i];
            })
        // blocks as list-of-lists
        .def_property("blocks",
            [](const Permutahedral_Simplex& s) {
                std::vector<std::vector<uint8_t>> out;
                for (int i = 0; i < s.num_blocks; ++i) {
                    std::vector<uint8_t> row(s.blocks[i], s.blocks[i] + s.block_sizes[i]);
                    out.push_back(row);
                },
                [](Permutahedral_Simplex& s, std::vector<std::vector<uint8_t>> v) {
                    for (int i = 0; i < (int)v.size() && i <= MAX_D; ++i)
                        for (int j = 0; j < (int)v[i].size() && j <= MAX_D; ++j)
                            s.blocks[i][j] = v[i][j];
                })
        .def("__eq__", &operator==)
        .def("__repr__", [](const Permutahedral_Simplex& s) {
            std::string r = "PermutahedralSimplex(dim=" + std::to_string(s.amb_dim)
                          + ", num_blocks=" + std::to_string(s.num_blocks) + ")";
            return r;
        });

    // -----------------------------------------------------------------------
    // FK_Triangulation
    // -----------------------------------------------------------------------
    py::class_<FK_Triangulation>(m, "FKTriangulation")
        .def(py::init<uint8_t>(), py::arg("d"))
        .def_readwrite("amb_dim", &FK_Triangulation::amb_dim)
        .def_readwrite("scale",   &FK_Triangulation::scale)
        // cartesian_coordinates: int32 array -> numpy float array
        .def("cartesian_coordinates",
            [](const FK_Triangulation& fk, std::vector<int32_t> point) {
                if ((int)point.size() < fk.amb_dim)
                    throw std::runtime_error("point length must equal amb_dim");
                double out[MAX_D] = {};
                fk.cartesian_coordinates(point.data(), out);
                return double_to_np(out, fk.amb_dim);
            },
            py::arg("point"),
            "Convert an integer lattice point to Cartesian coordinates.");

    // -----------------------------------------------------------------------
    // C_Triangulation (Coxeter A_{d-1} lattice)
    // -----------------------------------------------------------------------
    py::class_<C_Triangulation, FK_Triangulation>(m, "CTriangulation")
        .def(py::init<uint8_t>(), py::arg("d"));

    // -----------------------------------------------------------------------
    // locate_simplex
    // -----------------------------------------------------------------------
    m.def("locate_simplex",
        [](const FK_Triangulation& fk, py::array_t<double> point) {
            if (point.size() < fk.amb_dim)
                throw std::runtime_error("point length must equal amb_dim");
            double pt[MAX_D] = {};
            np_to_double(point, pt, fk.amb_dim);
            return locate_simplex(fk, pt);
        },
        py::arg("fk"), py::arg("point"),
        "Locate the FK simplex that contains the given Cartesian point.");

    // -----------------------------------------------------------------------
    // faces / cofaces
    // -----------------------------------------------------------------------
    m.def("faces",
        [](const Permutahedral_Simplex& s, uint8_t k) {
            Permutahedral_Simplex buf[MAX_FACES];
            int n = faces(s, buf, k);
            std::vector<Permutahedral_Simplex> out(buf, buf + n);
            return out;
        },
        py::arg("simplex"), py::arg("k"),
        "Return all k-dimensional faces of the given simplex.");

    m.def("cofaces",
        [](const Permutahedral_Simplex& s, uint8_t l) {
            Permutahedral_Simplex buf[MAX_COFACES];
            int n = cofaces(s, buf, l);
            std::vector<Permutahedral_Simplex> out(buf, buf + n);
            return out;
        },
        py::arg("simplex"), py::arg("l"),
        "Return all l-dimensional cofaces of the given simplex.");

    // -----------------------------------------------------------------------
    // edge_intersection
    // -----------------------------------------------------------------------
    m.def("edge_intersection",
        [](Permutahedral_Simplex s,
           const FK_Triangulation& fk,
           std::function<double(py::array_t<double>)> sdf_py)
        -> py::object
        {
            // Wrap the Python SDF callable into the C signature
            auto sdf_c = [&](const double* p) -> double {
                py::array_t<double> arr(fk.amb_dim, p);
                return sdf_py(arr);
            };

            double intersection_point[MAX_D] = {};
            bool hit = edge_intersection(s, fk, sdf_c, intersection_point);
            if (!hit) return py::none();
            return double_to_np(intersection_point, fk.amb_dim);
        },
        py::arg("simplex"), py::arg("fk"), py::arg("sdf"),
        R"doc(
Find the SDF zero-crossing on a 1-simplex (edge).

Parameters
----------
simplex : PermutahedralSimplex
    Must have num_blocks == 2 (i.e. be an edge).
fk : FKTriangulation
    The ambient triangulation.
sdf : callable (numpy.ndarray) -> float
    Signed distance function evaluated in Cartesian space.

Returns
-------
numpy.ndarray or None
    Intersection point in Cartesian space, or None if no crossing exists.
)doc");

    // -----------------------------------------------------------------------
    // traceManifold
    // -----------------------------------------------------------------------
    m.def("trace_manifold",
        [](const FK_Triangulation& fk,
           std::function<double(py::array_t<double>)> sdf_py,
           py::array_t<double> seed)
        -> std::vector<py::array_t<double>>
        {
            auto sdf_c = [&](const double* p) -> double {
                py::array_t<double> arr(fk.amb_dim, p);
                return sdf_py(arr);
            };

            double seed_c[MAX_D] = {};
            np_to_double(seed, seed_c, fk.amb_dim);

            std::unordered_map<Permutahedral_Simplex, Point, Permutahedral_Simplex_Hash> Ls;
            traceManifold(fk, sdf_c, Ls, seed_c);

            // Return as a flat list of intersection points
            std::vector<py::array_t<double>> out;
            out.reserve(Ls.size());
            for (const auto& [edge, pt] : Ls)
                out.push_back(double_to_np(pt.data(), fk.amb_dim));
            return out;
        },
        py::arg("fk"), py::arg("sdf"), py::arg("seed"),
        R"doc(
Trace the (d-1)-manifold defined by sdf=0 via BFS over the FK triangulation.

Parameters
----------
fk : FKTriangulation
    The ambient triangulation.
sdf : callable (numpy.ndarray) -> float
    Signed distance function.
seed : numpy.ndarray
    A Cartesian point near the zero-set to start BFS from.

Returns
-------
list of numpy.ndarray
    All found intersection points (one per intersected edge).
)doc");

    // -----------------------------------------------------------------------
    // triangulate_surface
    // -----------------------------------------------------------------------
    m.def("triangulate_surface",
        [](FK_Triangulation& fk,
           std::vector<std::pair<Permutahedral_Simplex, py::array_t<double>>> edge_points)
        -> std::vector<std::pair<Permutahedral_Simplex, std::vector<py::array_t<double>>>>
        {
            // Build the Ls map from Python input
            std::unordered_map<Permutahedral_Simplex, Point, Permutahedral_Simplex_Hash> Ls;
            for (auto& [edge, arr] : edge_points) {
                Point pt;
                auto r = arr.unchecked<1>();
                for (int i = 0; i < fk.amb_dim; ++i) pt[i] = r(i);
                Ls[edge] = pt;
            }

            std::unordered_map<
                Permutahedral_Simplex,
                std::vector<Point>,
                Permutahedral_Simplex_Hash
            > Ps;
            triangulate_surface(fk, Ls, Ps);

            // Convert back to Python
            std::vector<std::pair<Permutahedral_Simplex, std::vector<py::array_t<double>>>> out;
            for (auto& [simp, pts] : Ps) {
                std::vector<py::array_t<double>> py_pts;
                for (auto& p : pts)
                    py_pts.push_back(double_to_np(p.data(), fk.amb_dim));
                out.push_back({simp, py_pts});
            }
            return out;
        },
        py::arg("fk"), py::arg("edge_points"),
        R"doc(
Group edge intersection points into d-simplices for surface triangulation.

Parameters
----------
fk : FKTriangulation
edge_points : list of (PermutahedralSimplex, numpy.ndarray)
    Pairs of (edge, intersection_point) as returned by trace_manifold
    (use zip with the raw edge map if you need per-edge access).

Returns
-------
list of (PermutahedralSimplex, list of numpy.ndarray)
    Each entry is a d-simplex paired with the intersection points it owns.
)doc");
}