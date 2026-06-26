#pragma once

/*
Manifold Tracing implementation for d-1 manifolds using the Permutahedral 
Representation and finding intersection with 1-simplex

Aayush Rath
*/

#include "utils.h"
#include "permutahedral_simplex.h"
#include "fk_triangulation.h"
#include <cassert>
#include <unordered_map>
#include <queue>

using Point = std::array<double, MAX_D>;

bool edge_intersection(
    Permutahedral_Simplex& s,
    const FK_Triangulation& fk,
    double (*sdf)(const double*),
    double* intersection_point
) {
    assert(s.num_blocks == 2);

    int32_t v0[MAX_D];
    for (int i = 0; i < s.amb_dim; ++i) v0[i] = s.anchor[i];

    int32_t v1[MAX_D];
    for (int i = 0; i < s.amb_dim; ++i) v1[i] = s.anchor[i];

    for (int j = 0; j < s.block_sizes[0]; ++j) {
        int idx = s.blocks[0][j];
        if (idx == s.amb_dim) {
            for (int k = 0; k < s.amb_dim; ++k) v1[k]++;
        } else {
            v1[idx]++;
        }
    }

    // Cartesian coordinates
    double p0[MAX_D], p1[MAX_D];
    fk.cartesian_coordinates(v0, p0);
    fk.cartesian_coordinates(v1, p1);

    double f0 = sdf(p0);
    double f1 = sdf(p1);

    const double vertex_eps = 1e-10;
    if (std::abs(f0) < vertex_eps || std::abs(f1) < vertex_eps) {
        return false;
    }

    if (f0 * f1 >= 0.0) return false;

    Eigen::Matrix2d A;
    A << 1.0, 1.0,
         f0,  f1;

    Eigen::Vector2d b(1.0, 0.0);
    Eigen::Vector2d lambda = A.colPivHouseholderQr().solve(b);

    if (lambda(0)  < 1e-2 || lambda(1) < 1e-2) std::cout << "Lambda1: " << lambda(0) << " Lambda2: " << lambda(1) << std::endl;

    // Validate the solution
    const double bary_eps = 1e-8;
    const double residual_eps = 1e-8;

    // Check if solution is numerically valid
    Eigen::Vector2d residual = A * lambda - b;
    if (residual.norm() > residual_eps) {
        // Solution is not accurate enough
        return false;
    }

    // Check barycentric coordinates are in (0, 1) - strictly interior
    if (lambda(0) <= bary_eps || lambda(0) >= 1.0 - bary_eps ||
        lambda(1) <= bary_eps || lambda(1) >= 1.0 - bary_eps) {
        // Intersection is too close to a vertex
        return false;
    }

    // Check sum = 1
    if (std::abs(lambda.sum() - 1.0) > bary_eps) {
        return false;
    }

    // Compute intersection point
    for (int i = 0; i < fk.amb_dim; ++i) {
        intersection_point[i] = lambda(0) * p0[i] + lambda(1) * p1[i];
    }

    return true;
}

bool bound_check(
    const FK_Triangulation& fk,
    double* point
) {
    for (int i = 0; i < fk.amb_dim; i++) if (point[i] > 3.14 || point[i] < -3.14) return false;
    return true;
}

void traceManifold(
    const FK_Triangulation& fk,
    double (*sdf)(const double*),
    std::unordered_map<Permutahedral_Simplex, Point, Permutahedral_Simplex_Hash>& Ls,
    double* seed
) {
    std::queue<Permutahedral_Simplex> Q;

    Permutahedral_Simplex initial_simplex = locate_simplex(fk, seed);

    Permutahedral_Simplex initial_edges[MAX_FACES];
    int num_faces = faces(initial_simplex, initial_edges, 1);
    std::cout << num_faces << std::endl;

    for (int i = 0; i < num_faces; i++) {
        Q.push(initial_edges[i]);
        double intersection_point[MAX_D];
        bool h = edge_intersection(initial_edges[i], fk, sdf, intersection_point);
        Point p;
        for (int i = 0; i < fk.amb_dim; i++) p[i] = intersection_point[i];
        if (h) Ls[initial_edges[i]] = p;
    }

    int intersect_count =  0;

    while (!Q.empty()) {
        Permutahedral_Simplex edge = Q.front();
        Q.pop();

        Permutahedral_Simplex triangles [MAX_COFACES];
        int num_cofaces = cofaces(edge, triangles, 2);
        for (int i = 0; i < num_cofaces; i++) {
            Permutahedral_Simplex new_edges[MAX_FACES];
            int num_edges = faces(triangles[i], new_edges, 1);
            for (int j = 0; j < num_edges; j++) {
                double intersection_point[MAX_D];
                if (Ls.find(new_edges[j]) != Ls.end()) continue;
                if (!edge_intersection(new_edges[j], fk, sdf, intersection_point)) continue;
                // if (!bound_check(fk, intersection_point)) continue;
                intersect_count++;
                Q.push(new_edges[j]);
                // std::cout << "Intersection Point: ";
                // for(int i = 0; i < fk.amb_dim; i++) std::cout << intersection_point[i] << " ";
                // std::cout << std::endl;
                Point p;
                for (int i = 0; i < fk.amb_dim; i++)
                    p[i] = intersection_point[i];

                Ls[new_edges[j]] = p;
                // if (intersect_count % 10000 == 0)
                // {std::cout << "Intersection Point: ";
                // for(int i = 0; i < fk.amb_dim; i++) std::cout << Ls[new_edges[j]][i] << " ";
                // std::cout << std::endl;}
            }
        }

        if (intersect_count > 10) {
            std::cout << "Size of the queue: " << Q.size() << std::endl;
            break;
        }
    }

    std::cout << "Intersection Count: " << intersect_count << std::endl;
}

bool same_point(
    const Point& a,
    const Point& b,
    double eps = 1e-2
) {
    for (int i = 0; i < 3; i++) {
        if (std::abs(a[i] - b[i]) > eps)
            return false;
    }
    return true;
}


void triangulate_surface(
    FK_Triangulation& fk,
    const std::unordered_map<
        Permutahedral_Simplex,
        Point,
        Permutahedral_Simplex_Hash
    >& Ls,
    std::unordered_map<
        Permutahedral_Simplex,
        std::vector<Point>,
        Permutahedral_Simplex_Hash
    >& Ps
) {
    std::cout << "=== TRIANGULATE SURFACE (POINT-DEDUP TEST) ===" << std::endl;
    std::cout << "Input edges: " << Ls.size() << std::endl;
    int num_tri = 0;

    // Accumulate with edge-level deduplication
    for (const auto& [edge, point] : Ls) {
        Permutahedral_Simplex d_simplices[MAX_COFACES];
        int num_cofaces = cofaces(edge, d_simplices, fk.amb_dim);

        for (int i = 0; i < num_cofaces; i++) {
            if (!Ps[d_simplices[i]].empty())  {
                bool found_duplicate = false;
                for (auto& p : Ps[d_simplices[i]]) {
                    if (same_point(p, point)) {
                        found_duplicate = true;
                        break;
                    }
                }
                if (!found_duplicate) {
                    Ps[d_simplices[i]].push_back(point);
                }
            } else Ps[d_simplices[i]].push_back(point);
        }
    }

    for (auto& d_simp : Ps) {
        if (d_simp.second.size() < 3) num_tri++;
    }

    std::cout << "Number of triangles: " << num_tri << std::endl;
}
