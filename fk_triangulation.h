#pragma once 

/*
Freudenthal Kuhn Triangulation
for the ambient space (ℝ^d)

Aayush Rath
*/

#include "utils.h"
#include "permutahedral_simplex.h"
#include <Eigen/Dense>

struct FK_Triangulation {
    uint8_t amb_dim;                                                                                    // Ambient space dimension
    double scale = 1.0;                                                                                 // Lattice scaling

    double Lambda[MAX_D][MAX_D];                                                                        // Rotation matrix of skewing the latticr
    double Lambda_inv[MAX_D][MAX_D];                                                                    // Inverse to deskew
    double b[MAX_D];                                                                                    // Offset translation

    // Constructor for the FK Triangulation
    FK_Triangulation(uint8_t d) {
        this->amb_dim = d;
        this->scale = 1.0;
        for (int i = 0; i < amb_dim; i++) {                                                             // Initialize with an identity rotation matrix
            for (int j = 0; j < amb_dim; j++) {
                if (i == j) Lambda[i][j] = 1.0;
                else Lambda[i][j] = 0.0;
            }
        }

        for (int i = 0; i < amb_dim; i++) {
            for (int j = 0; j < amb_dim; j++) {
                if (i == j) Lambda_inv[i][j] = 1.0;
                else Lambda_inv[i][j] = 0.0;
            }
        }

        for (int i = 0;i < d; i++) b[i] = 0.0;
    }

    FK_Triangulation(uint8_t d, const Eigen::MatrixXd& matrix) : amb_dim(d) {
        Eigen::MatrixXd matrix_inv = matrix.inverse();
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                Lambda[i][j] = matrix(i, j);
                Lambda_inv[i][j] = matrix_inv(i, j);
            }
        }
    }

    // Convert the interger point to the corresponding Euclidean Space point coordinates
    void cartesian_coordinates(const int32_t *point, double *cartesian_point) const {
        for (int i = 0; i < amb_dim; i++) {
            cartesian_point[i] = 0.0;
            for (int j = 0; j < amb_dim; j++) {
                cartesian_point[i] += Lambda[i][j] * (point[j] / scale);
            }
            cartesian_point[i] += b[i];
        } 
    }
};

struct C_Triangulation : public FK_Triangulation {
    C_Triangulation(uint8_t d)
        : FK_Triangulation(d, compute_coxeter_matrix(d)) {}

private:
    static Eigen::MatrixXd compute_coxeter_matrix(int d) {
        using Matrix = Eigen::MatrixXd;
        
        // Build the Cartan matrix for type A_{d-1}
        Matrix cartan = Matrix::Identity(d, d);
        for (int i = 1; i < d; i++) {
            cartan(i - 1, i) = -0.5;
            cartan(i, i - 1) = -0.5;
        }
        
        // Compute the eigendecomposition
        Eigen::SelfAdjointEigenSolver<Matrix> saes(cartan);
        Matrix V = saes.eigenvectors();
        
        // Compute sqrt of eigenvalues
        Eigen::VectorXd sqrt_diag(d);
        for (int i = 0; i < d; i++) {
            sqrt_diag(i) = std::sqrt(saes.eigenvalues()[i]);
        }
        
        // The embedding matrix
        Matrix lower = Matrix::Ones(d, d).triangularView<Eigen::Lower>();
        Matrix result = (lower * V * sqrt_diag.asDiagonal()).inverse();
        
        return result;
    }
};

Permutahedral_Simplex locate_simplex(
    const FK_Triangulation& fk,
    const double *point
) {
    Permutahedral_Simplex s;
    s.amb_dim = fk.amb_dim;

    double x[MAX_D+1];                                                                                // The point transformed in the FK coordinate system
    double frac[MAX_D+1];                                                                             // Fractional part of the transformed point

    for (int i = 0; i < fk.amb_dim; i++) {
        double v = point[i] - fk.b[i];
        x[i] = 0.0;
        for (int j = 0; j < fk.amb_dim; j++) x[i] += fk.Lambda_inv[i][j] * v;
        x[i] *= fk.scale;
    }

    for (int i = 0; i < fk.amb_dim; i++) {
        int yi = (int)floor(x[i]);                                                                     // The interger points for the simplex anchor 
        s.anchor[i] = yi;
        frac[i] = x[i] - yi;
    }

    frac[fk.amb_dim] = std::numeric_limits<double>::infinity();
    
    uint8_t idx[MAX_D];

    // Set the idx to be the set {0, 1, ..., d}
    for (int i = 0; i <= fk.amb_dim; i++) idx[i] = i;

    // Sort the set according to the ascending order of the fractional part
    std::sort(idx, idx + fk.amb_dim + 1,
        [&frac](int i, int j) {
            return frac[i] < frac[j];
        });

    const double eps = 1e-12;
    s.num_blocks = 0;

    // Use the traversal order to set the ordered partition
    for (int i = 0; i <= fk.amb_dim; i++) {
        if (i == 0 || frac[idx[i]] - frac[idx[i-1]] > eps) {
            s.block_sizes[s.num_blocks] = 0;
            s.num_blocks++;
        }

        s.blocks[s.num_blocks - 1][s.block_sizes[s.num_blocks - 1]] = idx[i];
        s.block_sizes[s.num_blocks - 1]++;
    }

    return s;
}