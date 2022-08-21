#ifndef GRAPHICS_MATH_H_INCLUDE
#define GRAPHICS_MATH_H_INCLUDE

#include <array>
#include <string>
#include <sstream>
#include <cmath>

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

// C-like, basic function

template<typename T>
inline void print_vec_n(T* vec, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << vec[i] << ' ';
    }
    std::cout << '\n';
}

template<typename T>
inline void fill_vec_n(T* vec, T val, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = val;
    }
}

template<typename T>
inline std::string vec_to_string_n(T* vec, int n) {
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


template<typename T>
inline void add_vec_n(T* lin, T* rin, T* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = lin[i] + rin[i];
    }
}

template<typename T>
inline void add_scale_vec_n(T* in, T* out, T scale, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] + scale;
    }
}

template<typename T>
inline void sub_vec_n(T* lin, T* rin, T* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = lin[i] - rin[i];
    }
}

template<typename T>
inline void sub_scale_vec_n(T* in, T* out, T scale, int n, bool invert) {
    for (int i = 0; i < n; i++) {
        T val = in[i] - scale;
        if (invert) {
            val = -val;
        }
        out[i] = val;
    }
}

template<typename T>
inline void mul_vec_n(T* lin, T* rin, T* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = lin[i] * rin[i];
    }
}

template<typename T>
inline void mul_scale_vec_n(T* in, T* out, T scale, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = scale * in[i];
    }
}

template<typename T>
inline void div_vec_n(T* lin, T* rin, T* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = lin[i] / rin[i];
    }
}

template<typename T>
inline void div_scale_vec_n(T* in, T* out, T scale, int n, bool invert) {
    for (int i = 0; i < n; i++) {
        T val = in[i];
        if (!invert) {
            val = val/scale;
        } else {
            val = scale/val;
        }
        out[i] = val;
    }
}

template<typename T>
inline void cross_product_vec3(T* lin, T* rin, T* out) {
    out[0] = lin[1] * rin[2] - lin[2] * rin[1];
    out[1] = lin[2] * rin[0] - lin[0] * rin[2];
    out[2] = lin[0] * rin[1] - lin[1] * rin[0];
}

template<typename T>
inline void normalize_n(T* vec, int n) {
    double div = 0.f;
    for (int i = 0; i < n; i++) {
        T val = vec[i];
        div += val * val;
    }
    div = std::sqrt(div);

    for (int i = 0; i < n; i++) {
        vec[i] /= div;
    }
}

template<typename T>
inline T dot_n(T* lin, T* rin, int n) {
    double val = 0.f;
    for (int i = 0; i < n; i++) {
        val += lin[i] * rin[i];
    }
    return val;
}

template<typename T>
inline void mul_mat_n(T* lin, T* rin, T* out, int n) {
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

template<typename T>
inline void fill_diagonal_n(T* mat, T val, int n) {
    for (int i = 0; i < n; i++) {
        mat[i * n + i] = val;
    }
}

template<typename T>
inline void invert_mat_4(T* inv, T* mat4) {
    inv[0] = mat4[5]  * mat4[10] * mat4[15] -
             mat4[5]  * mat4[11] * mat4[14] -
             mat4[9]  * mat4[6]  * mat4[15] +
             mat4[9]  * mat4[7]  * mat4[14] +
             mat4[13] * mat4[6]  * mat4[11] -
             mat4[13] * mat4[7]  * mat4[10];

    inv[4] = -mat4[4]  * mat4[10] * mat4[15] +
              mat4[4]  * mat4[11] * mat4[14] +
              mat4[8]  * mat4[6]  * mat4[15] -
              mat4[8]  * mat4[7]  * mat4[14] -
              mat4[12] * mat4[6]  * mat4[11] +
              mat4[12] * mat4[7]  * mat4[10];

    inv[8] = mat4[4]  * mat4[9]  * mat4[15] -
             mat4[4]  * mat4[11] * mat4[13] -
             mat4[8]  * mat4[5]  * mat4[15] +
             mat4[8]  * mat4[7]  * mat4[13] +
             mat4[12] * mat4[5]  * mat4[11] -
             mat4[12] * mat4[7]  * mat4[9];

    inv[12] = -mat4[4]  * mat4[9]  * mat4[14] +
               mat4[4]  * mat4[10] * mat4[13] +
               mat4[8]  * mat4[5]  * mat4[14] -
               mat4[8]  * mat4[6]  * mat4[13] -
               mat4[12] * mat4[5]  * mat4[10] +
               mat4[12] * mat4[6]  * mat4[9];

    inv[1] = -mat4[1]  * mat4[10] * mat4[15] +
              mat4[1]  * mat4[11] * mat4[14] +
              mat4[9]  * mat4[2]  * mat4[15] -
              mat4[9]  * mat4[3]  * mat4[14] -
              mat4[13] * mat4[2]  * mat4[11] +
              mat4[13] * mat4[3]  * mat4[10];

    inv[5] = mat4[0]  * mat4[10] * mat4[15] -
             mat4[0]  * mat4[11] * mat4[14] -
             mat4[8]  * mat4[2]  * mat4[15] +
             mat4[8]  * mat4[3]  * mat4[14] +
             mat4[12] * mat4[2]  * mat4[11] -
             mat4[12] * mat4[3]  * mat4[10];

    inv[9] = -mat4[0]  * mat4[9]  * mat4[15] +
              mat4[0]  * mat4[11] * mat4[13] +
              mat4[8]  * mat4[1]  * mat4[15] -
              mat4[8]  * mat4[3]  * mat4[13] -
              mat4[12] * mat4[1]  * mat4[11] +
              mat4[12] * mat4[3]  * mat4[9];

    inv[13] = mat4[0]  * mat4[9]  * mat4[14] -
              mat4[0]  * mat4[10] * mat4[13] -
              mat4[8]  * mat4[1]  * mat4[14] +
              mat4[8]  * mat4[2]  * mat4[13] +
              mat4[12] * mat4[1]  * mat4[10] -
              mat4[12] * mat4[2]  * mat4[9];

    inv[2] = mat4[1]  * mat4[6] * mat4[15] -
             mat4[1]  * mat4[7] * mat4[14] -
             mat4[5]  * mat4[2] * mat4[15] +
             mat4[5]  * mat4[3] * mat4[14] +
             mat4[13] * mat4[2] * mat4[7] -
             mat4[13] * mat4[3] * mat4[6];

    inv[6] = -mat4[0]  * mat4[6] * mat4[15] +
              mat4[0]  * mat4[7] * mat4[14] +
              mat4[4]  * mat4[2] * mat4[15] -
              mat4[4]  * mat4[3] * mat4[14] -
              mat4[12] * mat4[2] * mat4[7] +
              mat4[12] * mat4[3] * mat4[6];

    inv[10] = mat4[0]  * mat4[5] * mat4[15] -
              mat4[0]  * mat4[7] * mat4[13] -
              mat4[4]  * mat4[1] * mat4[15] +
              mat4[4]  * mat4[3] * mat4[13] +
              mat4[12] * mat4[1] * mat4[7] -
              mat4[12] * mat4[3] * mat4[5];

    inv[14] = -mat4[0]  * mat4[5] * mat4[14] +
               mat4[0]  * mat4[6] * mat4[13] +
               mat4[4]  * mat4[1] * mat4[14] -
               mat4[4]  * mat4[2] * mat4[13] -
               mat4[12] * mat4[1] * mat4[6] +
               mat4[12] * mat4[2] * mat4[5];

    inv[3] = -mat4[1] * mat4[6] * mat4[11] +
              mat4[1] * mat4[7] * mat4[10] +
              mat4[5] * mat4[2] * mat4[11] -
              mat4[5] * mat4[3] * mat4[10] -
              mat4[9] * mat4[2] * mat4[7] +
              mat4[9] * mat4[3] * mat4[6];

    inv[7] = mat4[0] * mat4[6] * mat4[11] -
             mat4[0] * mat4[7] * mat4[10] -
             mat4[4] * mat4[2] * mat4[11] +
             mat4[4] * mat4[3] * mat4[10] +
             mat4[8] * mat4[2] * mat4[7] -
             mat4[8] * mat4[3] * mat4[6];

    inv[11] = -mat4[0] * mat4[5] * mat4[11] +
               mat4[0] * mat4[7] * mat4[9] +
               mat4[4] * mat4[1] * mat4[11] -
               mat4[4] * mat4[3] * mat4[9] -
               mat4[8] * mat4[1] * mat4[7] +
               mat4[8] * mat4[3] * mat4[5];

    inv[15] = mat4[0] * mat4[5] * mat4[10] -
              mat4[0] * mat4[6] * mat4[9] -
              mat4[4] * mat4[1] * mat4[10] +
              mat4[4] * mat4[2] * mat4[9] +
              mat4[8] * mat4[1] * mat4[6] -
              mat4[8] * mat4[2] * mat4[5];

    double det = mat4[0] * inv[0] + mat4[1] * inv[4] + mat4[2] * inv[8] + mat4[3] * inv[12];
    det = 1.0f / det;

    for (int i = 0; i < 16; i++) {
        inv[i] *= det;
    }
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
inline VectorBase<T, N> operator/ (VectorBase<T, N> a, VectorBase<T, N> &b) {
    div_vec_n(GetPtr(a), GetPtr(b), GetPtr(a), N); // cover the a array
    return a;
}

template<typename T, size_t N>
inline VectorBase<T, N> operator/ (VectorBase<T, N> a, T scale) {
    div_scale_vec_n(GetPtr(a), GetPtr(a), scale, N, false); // cover the a array
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
    fill_diagonal_n(GetPtr(mat), scale, N);
}

template<typename T, size_t N>
inline void Normalize(VectorBase<T, N> &vec) {
    normalize_n(GetPtr(vec), N);
}

template<typename T, size_t N>
inline T Dot(VectorBase<T, N> &a, VectorBase<T, N> &b) {
    return dot_n(GetPtr(a), GetPtr(b), N);
}

template<typename T>
inline Vector3<T> CrossProduct(Vector3<T> &a, Vector3<T> &b) {
    Vector3<T> out;
    cross_product_vec3(GetPtr(a), GetPtr(b), GetPtr(out));
    return out;
}

template<typename T>
inline Matrix4<T> InvertMat4(Matrix4<T> mat4) {
    Matrix4<T> inv;
    invert_mat_4(GetPtr(inv), GetPtr(mat4));
    return inv;
}

template<typename T>
inline Matrix4<T> LookAt(Vector3<T> &eye, Vector3<T> &center, Vector3<T> &up) {
    Matrix4<T> Matrix;
    Vector3<T> X, Y, Z;

    Z = eye - center;
    Normalize(Z);

    Y = up;
    X = CrossProduct(Y, Z);
    Y = CrossProduct(Z, X);

    Normalize(X);
    Normalize(Y);

    // x
    Matrix[0][0] = X[0];
    Matrix[1][0] = X[1];
    Matrix[2][0] = X[2];

    X = -X;
    Matrix[3][0] = Dot(X, eye);

    // y
    Matrix[0][1] = Y[0];
    Matrix[1][1] = Y[1];
    Matrix[2][1] = Y[2];

    Y = -Y;
    Matrix[3][1] = Dot(Y, eye);

    // z
    Matrix[0][2] = Z[0];
    Matrix[1][2] = Z[1];
    Matrix[2][2] = Z[2];

    Z = -Z;
    Matrix[3][2] = Dot(Z, eye);

    // others
    Matrix[0][3] = 0;
    Matrix[1][3] = 0;
    Matrix[2][3] = 0;
    Matrix[3][3] = 1.0f;

    return Matrix;
}

#endif
