#ifndef GRAPHICS_MATH_H_INCLUDE
#define GRAPHICS_MATH_H_INCLUDE

#include <array>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib>

// basic types

template<typename T, size_t N>
using VectorBase = std::array<T, N>;

template<typename T, size_t N>
using MatrixBase = std::array<std::array<T, N>, N>;

// for Euclidean spaces types

template<typename T>
using Vector3 = VectorBase<T, 3>;

using Vector3f = VectorBase<float, 3>;

template<typename T>
using Matrix4 = MatrixBase<T, 4>;

using Matrix4f = MatrixBase<float, 4>;

// C-like basic functions. We need to provide the size if the postfix is '_n'. For example,
// print_vec_n(). Only give the fixed size array if the postfix is number. For example,
// crossproduct_vec3.


// Print the n size vector elements.
template<typename T>
inline void print_vec_n(const T* vec, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << vec[i] << ' ';
    }
    std::cout << '\n';
}

// Fill the n size vector.
template<typename T>
inline void fill_vec_n(T* vec, T scale, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = scale;
    }
}

// Convert the n size vector to std::string.
template<typename T>
inline std::string vec_to_string_n(const T* vec, int n) {
    auto out = std::ostringstream{};
    out << '(';
    for (int i = 0; i < n; i++) {
        out << vec[i];
        if (i != n-1) {
            out << ", ";
        }
    }
    out << ')';
    return out.str();
}

// Then n size vectors addition operation, [out vec] = [left vec] + [right vec].
template<typename T>
inline void add_vec_n(const T* lin, const T* rin, T* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = lin[i] + rin[i];
    }
}

// The n size vectors and scale addition operation, [out vec] = [left vec] + scale.
template<typename T>
inline void add_scale_vec_n(const T* in, T* out, T scale, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] + scale;
    }
}

// The n size vectors subtraction operation, [out vec] = [left vec] - [right vec].
template<typename T>
inline void sub_vec_n(const T* lin, const T* rin, T* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = lin[i] - rin[i];
    }
}

// The n size vectors and scale subtraction operation, 
//     [out vec] = [left vec] - scale,      if not invert.
//     [out vec] =      scale - [left vec], if invert.
template<typename T>
inline void sub_scale_vec_n(const T* in, T* out, T scale, int n, bool invert) {
    for (int i = 0; i < n; i++) {
        T val = in[i] - scale;
        if (invert) {
            val = -val;
        }
        out[i] = val;
    }
}

// The n size vectors multiplication operation, [out vec] = [left vec] x [right vec].
template<typename T>
inline void mul_vec_n(const T* lin, const T* rin, T* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = lin[i] * rin[i];
    }
}

// The n size vectors and scale multiplication operation, [out vec] = [left vec] x scale.
template<typename T>
inline void mul_scale_vec_n(const T* in, T* out, T scale, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = scale * in[i];
    }
}

// The n size vectors division operation, [out vec] = [left vec] / [right vec].
// This operation is not common.
template<typename T>
inline void div_vec_n(const T* lin, const T* rin, T* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = lin[i] / rin[i];
    }
}

// The n size vectors and scale division operation, 
//     [out vec] = [left vec] / scale,      if not invert.
//     [out vec] =      scale / [left vec], if invert.
template<typename T>
inline void div_scale_vec_n(const T* in, T* out, T scale, int n, bool invert) {
    for (int i = 0; i < n; i++) {
        T val = in[i];
        if (invert) {
            val = scale/val;
        } else {
            val = val/scale;
        }
        out[i] = val;
    }
}

// The 3-vector cross product operation.
// wiki: https://en.wikipedia.org/wiki/Cross_product
template<typename T>
inline void crossproduct_vec3(const T* lin, const T* rin, T* out) {
    out[0] = lin[1] * rin[2] - lin[2] * rin[1];
    out[1] = lin[2] * rin[0] - lin[0] * rin[2];
    out[2] = lin[0] * rin[1] - lin[1] * rin[0];
}

// The n size vector inner product operation.
// wiki: https://en.wikipedia.org/wiki/Inner_product_space
template<typename T>
inline T innerproduct_vec_n(const T* lin, const T* rin, int n) {
    double val = 0.f;
    for (int i = 0; i < n; i++) {
        val += lin[i] * rin[i];
    }
    return val;
}

// The  n-size vector normalizing operation. It should be
// equal to
//     |a| = sqrt(inner product(a, a))
//     out = a / |a|
template<typename T>
inline void normalize_vec_n(T* vec, int n) {
    double factor = 0.f;
    for (int i = 0; i < n; i++) {
        T val = vec[i];
        factor += val*val;
    }
    factor = 1.f/std::sqrt(factor);

    for (int i = 0; i < n; i++) {
        vec[i] *= factor;
    }
}

// The n*n size matrix multiplication operation
// wiki: https://en.wikipedia.org/wiki/Matrix_multiplication
template<typename T>
inline void mul_mat_n(const T* lin, const T* rin, T* out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            T sum = 0;
            for (int k = 0; k < n; k++) {
                sum += lin[i * n + k] * rin[k * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

// Fill a the diagonal elements for n*n size matrix.
template<typename T>
inline void diagonal_mat_n(T* mat, T scale, int n) {
    for (int i = 0; i < n; i++) {
        mat[i * n + i] = scale;
    }
}

// Compute the 1*1 size matrix or 2*2 size matrix determinant value.
template<typename T>
inline T determinant_mat_low(const T* mat, int n) {
    if (n == 1) {
        return mat[0];
    } else if (n == 2) {
        T a = mat[0];
        T b = mat[1];
        T c = mat[n];
        T d = mat[n+1];
        return a*d - c*b;
    }
    return 0;
}

// Get the cofactor matrix for n*n size matrix.
template<typename T>
void fill_cofactor_n(const T* mat, T* cof, int p, int q, int n) {
    int sub_cnt = 0, main_cnt = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != p && j != q) {
                cof[sub_cnt++] = mat[main_cnt];
            }
            main_cnt++;
        }
    }
}

// Compute the n*n size matrix determinant value.
// wiki: https://en.wikipedia.org/wiki/Determinant
template<typename T>
inline T determinant_mat_n(const T* mat, int n) {
    if (n <= 2) {
        return determinant_mat_low(mat, n);
    }
    T* cof = (T*)std::malloc(sizeof(T) * (n-1) * (n-1));
    T det = 0;

    for (int i = 0; i < n; i++) {
        fill_cofactor_n(mat, cof, 0, i, n);
        T val = mat[i] * determinant_mat_n(cof, n-1);

        if ((i + 0) % 2 == 1) {
            val = -val;
        }
        det += val;
    }

    free(cof);
    return det;
}

// Compute the adjugate matrix for n*n size matrix.
template<typename T>
inline void adjugate_mat_n(const T* mat, T* adj, int n) {
    T* cof = (T*)std::malloc(sizeof(T) * (n-1) * (n-1));

    if (n == 1) {
        adj[0] = mat[0];
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fill_cofactor_n(mat, cof, i, j, n);
            T det = determinant_mat_n(cof, n-1); 

            if ((i + j) % 2 == 1) {
                det = -det;
            }
            adj[j * n + i] = det;
        }
    }

    free(cof);
}

// Compute the inverse of the n*n matrix.
// wiki: https://en.wikipedia.org/wiki/Invertible_matrix
template<typename T>
inline void invert_mat_n(const T* mat, T* inv, int n) {
    T* adj = (T*)std::malloc(sizeof(T) * n * n);

    adjugate_mat_n(mat, adj, n);
    T det = determinant_mat_n(mat, n);

    div_scale_vec_n(adj, inv, det, n*n, false);

    free(adj);
}

// Fast compute the inverse of the 4x4 matrix.
template<typename T>
inline void invert_mat4(const T* mat4, T* inv4) {
    inv4[0] = mat4[5]  * mat4[10] * mat4[15] -
              mat4[5]  * mat4[11] * mat4[14] -
              mat4[9]  * mat4[6]  * mat4[15] +
              mat4[9]  * mat4[7]  * mat4[14] +
              mat4[13] * mat4[6]  * mat4[11] -
              mat4[13] * mat4[7]  * mat4[10];

    inv4[4] = -mat4[4]  * mat4[10] * mat4[15] +
               mat4[4]  * mat4[11] * mat4[14] +
               mat4[8]  * mat4[6]  * mat4[15] -
               mat4[8]  * mat4[7]  * mat4[14] -
               mat4[12] * mat4[6]  * mat4[11] +
               mat4[12] * mat4[7]  * mat4[10];

    inv4[8] = mat4[4]  * mat4[9]  * mat4[15] -
              mat4[4]  * mat4[11] * mat4[13] -
              mat4[8]  * mat4[5]  * mat4[15] +
              mat4[8]  * mat4[7]  * mat4[13] +
              mat4[12] * mat4[5]  * mat4[11] -
              mat4[12] * mat4[7]  * mat4[9];

    inv4[12] = -mat4[4]  * mat4[9]  * mat4[14] +
                mat4[4]  * mat4[10] * mat4[13] +
                mat4[8]  * mat4[5]  * mat4[14] -
                mat4[8]  * mat4[6]  * mat4[13] -
                mat4[12] * mat4[5]  * mat4[10] +
                mat4[12] * mat4[6]  * mat4[9];

    inv4[1] = -mat4[1]  * mat4[10] * mat4[15] +
               mat4[1]  * mat4[11] * mat4[14] +
               mat4[9]  * mat4[2]  * mat4[15] -
               mat4[9]  * mat4[3]  * mat4[14] -
               mat4[13] * mat4[2]  * mat4[11] +
               mat4[13] * mat4[3]  * mat4[10];

    inv4[5] = mat4[0]  * mat4[10] * mat4[15] -
              mat4[0]  * mat4[11] * mat4[14] -
              mat4[8]  * mat4[2]  * mat4[15] +
              mat4[8]  * mat4[3]  * mat4[14] +
              mat4[12] * mat4[2]  * mat4[11] -
              mat4[12] * mat4[3]  * mat4[10];

    inv4[9] = -mat4[0]  * mat4[9]  * mat4[15] +
               mat4[0]  * mat4[11] * mat4[13] +
               mat4[8]  * mat4[1]  * mat4[15] -
               mat4[8]  * mat4[3]  * mat4[13] -
               mat4[12] * mat4[1]  * mat4[11] +
               mat4[12] * mat4[3]  * mat4[9];

    inv4[13] = mat4[0]  * mat4[9]  * mat4[14] -
               mat4[0]  * mat4[10] * mat4[13] -
               mat4[8]  * mat4[1]  * mat4[14] +
               mat4[8]  * mat4[2]  * mat4[13] +
               mat4[12] * mat4[1]  * mat4[10] -
               mat4[12] * mat4[2]  * mat4[9];

    inv4[2] = mat4[1]  * mat4[6] * mat4[15] -
              mat4[1]  * mat4[7] * mat4[14] -
              mat4[5]  * mat4[2] * mat4[15] +
              mat4[5]  * mat4[3] * mat4[14] +
              mat4[13] * mat4[2] * mat4[7] -
              mat4[13] * mat4[3] * mat4[6];

    inv4[6] = -mat4[0]  * mat4[6] * mat4[15] +
               mat4[0]  * mat4[7] * mat4[14] +
               mat4[4]  * mat4[2] * mat4[15] -
               mat4[4]  * mat4[3] * mat4[14] -
               mat4[12] * mat4[2] * mat4[7] +
               mat4[12] * mat4[3] * mat4[6];

    inv4[10] = mat4[0]  * mat4[5] * mat4[15] -
               mat4[0]  * mat4[7] * mat4[13] -
               mat4[4]  * mat4[1] * mat4[15] +
               mat4[4]  * mat4[3] * mat4[13] +
               mat4[12] * mat4[1] * mat4[7] -
               mat4[12] * mat4[3] * mat4[5];

    inv4[14] = -mat4[0]  * mat4[5] * mat4[14] +
                mat4[0]  * mat4[6] * mat4[13] +
                mat4[4]  * mat4[1] * mat4[14] -
                mat4[4]  * mat4[2] * mat4[13] -
                mat4[12] * mat4[1] * mat4[6] +
                mat4[12] * mat4[2] * mat4[5];

    inv4[3] = -mat4[1] * mat4[6] * mat4[11] +
               mat4[1] * mat4[7] * mat4[10] +
               mat4[5] * mat4[2] * mat4[11] -
               mat4[5] * mat4[3] * mat4[10] -
               mat4[9] * mat4[2] * mat4[7] +
               mat4[9] * mat4[3] * mat4[6];

    inv4[7] = mat4[0] * mat4[6] * mat4[11] -
              mat4[0] * mat4[7] * mat4[10] -
              mat4[4] * mat4[2] * mat4[11] +
              mat4[4] * mat4[3] * mat4[10] +
              mat4[8] * mat4[2] * mat4[7] -
              mat4[8] * mat4[3] * mat4[6];

    inv4[11] = -mat4[0] * mat4[5] * mat4[11] +
                mat4[0] * mat4[7] * mat4[9] +
                mat4[4] * mat4[1] * mat4[11] -
                mat4[4] * mat4[3] * mat4[9] -
                mat4[8] * mat4[1] * mat4[7] +
                mat4[8] * mat4[3] * mat4[5];

    inv4[15] = mat4[0] * mat4[5] * mat4[10] -
               mat4[0] * mat4[6] * mat4[9] -
               mat4[4] * mat4[1] * mat4[10] +
               mat4[4] * mat4[2] * mat4[9] +
               mat4[8] * mat4[1] * mat4[6] -
               mat4[8] * mat4[2] * mat4[5];

    double det = mat4[0] * inv4[0] + mat4[1] * inv4[4] + mat4[2] * inv4[8] + mat4[3] * inv4[12];
    det = 1.0f / det;

    for (int i = 0; i < 16; i++) {
        inv4[i] *= det;
    }
}

// Compute the look-at matrix.
template<typename T>
inline void lookat_mat4(const T* eye, const T* center, const T* up, T* mat4) {
    T* buf_x = (T*)std::malloc(sizeof(T) * 3);
    T* buf_y = (T*)std::malloc(sizeof(T) * 3);
    T* buf_z = (T*)std::malloc(sizeof(T) * 3);

    sub_vec_n(eye, center, buf_z, 3);
    normalize_vec_n(buf_z, 3);

    crossproduct_vec3(up   , buf_z, buf_x);
    crossproduct_vec3(buf_z, buf_x, buf_y);

    normalize_vec_n(buf_x, 3);
    normalize_vec_n(buf_y, 3);

    for (int i = 0; i < 3; ++i) {
        mat4[i * 4 + 0] = buf_x[i];
        mat4[i * 4 + 1] = buf_y[i];
        mat4[i * 4 + 2] = buf_z[i];

        buf_x[i] = -buf_x[i];
        buf_y[i] = -buf_y[i];
        buf_z[i] = -buf_z[i];
    }

    mat4[3 * 4 + 0] = innerproduct_vec_n(buf_x, eye, 3);
    mat4[3 * 4 + 1] = innerproduct_vec_n(buf_y, eye, 3);
    mat4[3 * 4 + 2] = innerproduct_vec_n(buf_z, eye, 3);

    mat4[0 * 4 + 3] = 0;
    mat4[1 * 4 + 3] = 0;
    mat4[2 * 4 + 3] = 0;
    mat4[3 * 4 + 3] = 1.f;

    free(buf_x);
    free(buf_y);
    free(buf_z);
}

template<typename T, size_t N>
inline T* get_vec_ptr(VectorBase<T, N> &vec) {
    return vec.data();
}

template<typename T, size_t N>
inline T* get_mat_ptr(MatrixBase<T, N> &mat) {
    return mat.at(0).data();
}

template<typename T, size_t N>
inline T* GetPtr(VectorBase<T, N> &vec) {
    return get_vec_ptr(vec);
}

template<typename T, size_t N>
inline T* GetPtr(MatrixBase<T, N> &mat) {
    return get_mat_ptr(mat);
}

template<typename T, size_t N>
inline std::string ToString(VectorBase<T, N> vec) {
    auto out = std::ostringstream{};
    out << vec_to_string_n(GetPtr(vec), N);
    return out.str();
}

template<typename T, size_t N>
inline std::string ToString(MatrixBase<T, N> mat) {
    auto out = std::ostringstream{};
    out << '{';
    for (int i = 0; i < N; i++) {
        out << vec_to_string_n(GetPtr(mat)+i*N, N);
        if (i != N-1) {
            out << ", ";
        }
    }
    out << '}';
    return out.str();
}

/*

// operators for vector

template<typename T, size_t N>
inline VectorBase<T, N> operator+ (VectorBase<T, N> a, VectorBase<T, N> &b) {
    add_vec_n(GetPtr(a), GetPtr(b), GetPtr(a), N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator+ (VectorBase<T, N> a, T scale) {
    add_scale_vec_n(GetPtr(a), GetPtr(a), scale, N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator+ (T scale, VectorBase<T, N> a) {
    add_scale_vec_n(GetPtr(a), GetPtr(a), scale, N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator+= (VectorBase<T, N> &a, VectorBase<T, N> &b) {
    add_vec_n(GetPtr(a), GetPtr(b), GetPtr(a), N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator+= (VectorBase<T, N> &a, T scale) {
    add_scale_vec_n(GetPtr(a), GetPtr(a), scale, N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator- (VectorBase<T, N> a, VectorBase<T, N> &b) {
    sub_vec_n(GetPtr(a), GetPtr(b), GetPtr(a), N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator- (VectorBase<T, N> a, T scale) {
    sub_scale_vec_n(GetPtr(a), GetPtr(a), scale, N, false); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator-= (VectorBase<T, N> &a, VectorBase<T, N> &b) {
    sub_vec_n(GetPtr(a), GetPtr(b), GetPtr(a), N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator-= (VectorBase<T, N> &a, T scale) {
    sub_scale_vec_n(GetPtr(a), GetPtr(a), scale, N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator* (VectorBase<T, N> a, VectorBase<T, N> &b) {
    mul_vec_n(GetPtr(a), GetPtr(b), GetPtr(a), N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator* (VectorBase<T, N> a, T scale) {
    mul_scale_vec_n(GetPtr(a), GetPtr(a), scale, N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator* (T scale, VectorBase<T, N> a) {
    mul_scale_vec_n(GetPtr(a), GetPtr(a), scale, N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator/ (VectorBase<T, N> a, T scale) {
    div_scale_vec_n(GetPtr(a), GetPtr(a), scale, N, false); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator/ (T scale, VectorBase<T, N> a) {
    div_scale_vec_n(GetPtr(a), GetPtr(a), scale, N, true); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator- (VectorBase<T, N> a) {
    T scale = -1;
    mul_scale_vec_n(GetPtr(a), GetPtr(a), scale, N); // cover the a array
    return a;
}

// operators for matrix

template<typename T, size_t N>
inline MatrixBase<T, N> operator+ (MatrixBase<T, N> a, MatrixBase<T, N> &b) {
    add_vec_n(GetPtr(a), GetPtr(b), GetPtr(a), N*N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline MatrixBase<T, N> operator+ (MatrixBase<T, N> a, T scale) {
    add_scale_vec_n(GetPtr(a), GetPtr(a), scale, N*N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline MatrixBase<T, N> operator+ (T scale, MatrixBase<T, N> a) {
    add_scale_vec_n(GetPtr(a), GetPtr(a), scale, N*N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline MatrixBase<T, N> operator+= (MatrixBase<T, N> &a, MatrixBase<T, N> &b) {
    add_vec_n(GetPtr(a), GetPtr(b), GetPtr(a), N*N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline MatrixBase<T, N> operator+= (MatrixBase<T, N> &a, T scale) {
    add_scale_vec_n(GetPtr(a), GetPtr(a), scale, N*N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline MatrixBase<T, N> operator- (MatrixBase<T, N> a, MatrixBase<T, N> &b) {
    sub_vec_n(GetPtr(a), GetPtr(b), GetPtr(a), N*N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline MatrixBase<T, N> operator- (MatrixBase<T, N> a, T scale) {
    sub_scale_vec_n(GetPtr(a), GetPtr(a), scale, N*N, false); // cover the a array
    return a;
}

template<typename T, size_t N>
inline MatrixBase<T, N> operator-= (MatrixBase<T, N> &a, MatrixBase<T, N> &b) {
    sub_vec_n(GetPtr(a), GetPtr(b), GetPtr(a), N*N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline MatrixBase<T, N> operator-= (MatrixBase<T, N> &a, T scale) {
    sub_scale_vec_n(GetPtr(a), GetPtr(a), scale, N*N, false); // cover the a array
    return a;
}

template<typename T, size_t N>
inline MatrixBase<T, N> operator* (MatrixBase<T, N> &a, MatrixBase<T, N> &b) {
    MatrixBase<T, N> out;
    mul_mat_n(GetPtr(a), GetPtr(b), GetPtr(out), N);
    return out;
}

template<typename T, size_t N>
inline MatrixBase<T, N> operator* (MatrixBase<T, N> a, T scale) {
    mul_scale_vec_n(GetPtr(a), GetPtr(a), scale, N*N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline MatrixBase<T, N> operator* (T scale, MatrixBase<T, N> a) {
    mul_scale_vec_n(GetPtr(a), GetPtr(a), scale, N*N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline MatrixBase<T, N> operator- (MatrixBase<T, N> a) {
    T scale = -1;
    mul_scale_vec_n(GetPtr(a), GetPtr(a), scale, N*N); // cover the a array
    return a;
}

// for graphic math

template<typename T, size_t N>
inline void FillScale(VectorBase<T, N> &vec, T scale) {
    fill_vec_n(GetPtr(vec), scale, N);
}

template<typename T, size_t N>
inline void FillScale(MatrixBase<T, N> &mat, T scale) {
    fill_vec_n(GetPtr(mat), scale, N*N);
}

template<typename T, size_t N>
inline void FillDiagonal(MatrixBase<T, N> &mat, T scale) {
    fill_vec_n(GetPtr(mat), (T)0, N*N);
    diagonal_mat_n(GetPtr(mat), scale, N);
}

template<typename T, size_t N>
inline void Normalize(VectorBase<T, N> &vec) {
    normalize_vec_n(GetPtr(vec), N);
}

template<typename T, size_t N>
inline T Dot(VectorBase<T, N> &a, VectorBase<T, N> &b) {
    return innerproduct_vec_n(GetPtr(a), GetPtr(b), N);
}

template<typename T, size_t N>
inline T InnerProduct(VectorBase<T, N> &a, VectorBase<T, N> &b) {
    return innerproduct_vec_n(GetPtr(a), GetPtr(b), N);
}

template<typename T>
inline Vector3<T> CrossProduct(Vector3<T> &a, Vector3<T> &b) {
    Vector3<T> out;
    crossproduct_vec3(GetPtr(a), GetPtr(b), GetPtr(out));
    return out;
}

template<typename T>
inline Matrix4<T> InvertMat4(Matrix4<T> mat4) {
    Matrix4<T> inv;
    invert_mat4(GetPtr(mat4), GetPtr(inv));
    return inv;
}

template<typename T>
inline Matrix4<T> LookAt(Vector3<T> &eye, Vector3<T> &center, Vector3<T> &up) {
    Matrix4<T> Matrix;
    
    lookat_mat4(GetPtr(eye), GetPtr(center), GetPtr(up), GetPtr(Matrix));

    return Matrix;
}
*/

#endif
