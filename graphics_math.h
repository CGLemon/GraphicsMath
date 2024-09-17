#pragma

#define _USE_MATH_DEFINES // for math.h

#include <iostream>
#include <iomanip>

#include <array>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <initializer_list>

#define FOREACH_LOOP(N)    \
    do {                   \
        size_t i = 0;      \
        for (; i < N; ++i) \

#define FOREACH_LOOP_END   \
    } while (0)

#define CONST_SCALE_TYPE const double
#define SCALE_TYPE double

#define IS_NUMBER(T) \
    (std::is_floating_point<T>::value || std::is_integral<T>::value)
#define IS_SAME(T, U) \
    std::is_same<T, typename std::remove_reference<U>::type>::value
#define IS_BASE_OF(T, U) \
    std::is_base_of<T, typename std::remove_reference<U>::type>::value

// C-like functions

static constexpr int kAxisX = 0;
static constexpr int kAxisY = 1;
static constexpr int kAxisZ = 2;
static constexpr int kAxisW = 3;

// Print a format number.
template<typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
std::string ScaleToString(T scale) {
    auto out = std::ostringstream{};
    int p = 6;
    int w = p + 6;
    out << std::fixed
            << std::setw(w)
            << std::setprecision(p)
            << scale;
    return out.str();
};

// Convert the vector-N to std::string.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
inline std::string VecorNToString(const T* vec) {
    auto out = std::ostringstream{};
    out << '(';

    FOREACH_LOOP(N) {
        out << ScaleToString(vec[i])
                << (i != N-1 ? ", " : ")");
    } FOREACH_LOOP_END;

    return out.str();
}

// Print the vector-N elements.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
inline void PrintVectorN(const T* vec) {
    std::cout << VecorNToString<N>(vec) << std::endl;
}

// Convert the matrix-N to std::string.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
inline std::string MatrixNToString(const T* mat) {
    auto out = std::ostringstream{};
    for (int i = 0; i < N; ++i) {
        if (i == 0) {
            out << "{ ";
        } else {
            out << "  ";
        }

        const T* offset = mat + i * N;
        out << VecorNToString<N>(offset);
        if (i != N-1) {
            out << "\n";
        }
    }
    out << " }";
    return out.str();
}

// Print the matrix-N elements.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
inline void PrintMatrixN(const T* mat) {
    std::cout << MatrixNToString<N>(mat) << std::endl;
}

template<size_t N, typename T>
constexpr T GetMatrixNIndex(T x, T y) {
    return y * N + x;
}

template<size_t N, typename T>
constexpr T GetMatrixNIndex(T dim) {
    return GetMatrixNIndex<N>(dim, dim);
}

// Fill the vector-N as same element, [out vec] = scale.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void FillVectorN(T* vec, CONST_SCALE_TYPE scale) {
    FOREACH_LOOP(N) { vec[i] = scale; } FOREACH_LOOP_END;
}

// Fill the vector-N, [out vec] = [in vec].
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void FillVectorN(const T* in, T* out) {
    FOREACH_LOOP(N) { out[i] = in[i]; } FOREACH_LOOP_END;
}

// The vector-N addition operation, [out vec] = [left vec] + [right vec].
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void AddVectorN(const T* lhs, const T* rhs, T* out) {
    FOREACH_LOOP(N) { out[i] = lhs[i] + rhs[i]; } FOREACH_LOOP_END;
}

// The vector-N addition operation, [out vec] = [in vec] + scale.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void AddVectorN(const T* in, T* out, CONST_SCALE_TYPE scale) {
    FOREACH_LOOP(N) { out[i] = in[i] + scale; } FOREACH_LOOP_END;
}

// The vector-N subtraction operation, [out vec] = [left vec] - [right vec].
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void SubVectorN(const T* lhs, const T* rhs, T* out) {
    FOREACH_LOOP(N) { out[i] = lhs[i] - rhs[i]; } FOREACH_LOOP_END;
}

// The vector-N and scale subtraction operation,
//     [out vec] = [in vec] - scale,    if not invert.
//     [out vec] =    scale - [in vec], if invert.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void SubVectorN(const T* in, T* out, CONST_SCALE_TYPE scale, const bool invert) {
    if (invert) {
        FOREACH_LOOP(N) { out[i] = scale - in[i]; } FOREACH_LOOP_END;
    } else {
        FOREACH_LOOP(N) { out[i] = in[i] - scale; } FOREACH_LOOP_END;
    }
}

// The vector-N multiplication operation, [out vec] = [left vec] x [right vec].
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void MulVectorN(const T* lhs, const T* rhs, T* out) {
    FOREACH_LOOP(N) { out[i] = lhs[i] * rhs[i]; } FOREACH_LOOP_END;
}

// The vector-N multiplication operation, [in vec] = [in vec] * scale.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void MulVectorN(const T* in, T* out, CONST_SCALE_TYPE scale) {
    FOREACH_LOOP(N) { out[i] = in[i] * scale; } FOREACH_LOOP_END;
}

// The vector-N division operation, [out vec] = [left vec] / [right vec].
// This operation is not common.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void DivVectorN(const T* lhs, const T* rhs, T* out) {
    FOREACH_LOOP(N) { out[i] = lhs[i] / rhs[i]; } FOREACH_LOOP_END;
}

// The vector-N and scale division operation,
//     [out vec] = [in vec] / scale,    if not invert.
//     [out vec] =    scale / [in vec], if invert.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void DivVectorN(const T* in, T* out, CONST_SCALE_TYPE scale, const bool invert) {
    if (invert) {
        FOREACH_LOOP(N) { out[i] = scale / in[i]; } FOREACH_LOOP_END;
    } else {
        FOREACH_LOOP(N) { out[i] = in[i] / scale; } FOREACH_LOOP_END;
    }
}

// The vector-N cross product operation.
// wiki: https://en.wikipedia.org/wiki/Cross_product
template<size_t N, typename T, typename = std::enable_if_t<N == 3 && IS_NUMBER(T)>>
constexpr void CrossProductVectorN(const T* lhs, const T* rhs, T* out) {
    out[kAxisX] = lhs[kAxisY] * rhs[kAxisZ] - lhs[kAxisZ] * rhs[kAxisY];
    out[kAxisY] = lhs[kAxisZ] * rhs[kAxisX] - lhs[kAxisX] * rhs[kAxisZ];
    out[kAxisZ] = lhs[kAxisX] * rhs[kAxisY] - lhs[kAxisY] * rhs[kAxisX];
}

// The vector-N inner product operation.
// wiki: https://en.wikipedia.org/wiki/Inner_product_space
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr T InnerProductVectorN(const T* lhs, const T* rhs) {
    SCALE_TYPE scale = 0.;
    FOREACH_LOOP(N) { scale += lhs[i] * rhs[i]; } FOREACH_LOOP_END;
    return scale;
}

// The vector-N normalization operation. It should be
// equal to
//     |a| = sqrt(inner product(a, a))
//     out = a / |a|
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void NormalizeVectorN(const T* in, T* out) {
    MulVectorN<N>(in, out, 1.0/std::sqrt(InnerProductVectorN<N>(in, in)));
}

// Fill the matrix-N as same element, [out vec] = scale.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void FillMatrixN(T* mat, CONST_SCALE_TYPE scale) {
    FillVectorN<N * N>(mat, scale);
}

// Fill the diagonal elements for matrix-N.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void FillDiagonalMatrixN(T* mat, CONST_SCALE_TYPE scale) {
    FOREACH_LOOP(N) { mat[GetMatrixNIndex<N>(i)] = scale; } FOREACH_LOOP_END;
}

// The matrix-N addition operation, [out mat] = [left mat] + [right mat].
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void AddMatrixN(const T* lhs, const T* rhs, T* out) {
    AddVectorN<N * N>(lhs, rhs, out);
}

// The matrix-N addition operation, [out mat] = [in mat] + scale.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void AddMatrixN(const T* in, T* out, CONST_SCALE_TYPE scale) {
    AddVectorN<N * N>(in, out, scale);
}

// The matrix-N subtraction operation, [out mat] = [left mat] - [right mat].
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void SubMatrixN(const T* lhs, const T* rhs, T* out) {
    SubVectorN<N * N>(lhs, rhs, out);
}

// The matrix-N and scale subtraction operation,
//     [out mat] = [in mat] - scale,    if not invert.
//     [out mat] =    scale - [in mat], if invert.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void SubMatrixN(const T* in, T* out, CONST_SCALE_TYPE scale, const bool invert) {
    SubVectorN<N * N>(in, out, scale, invert);
}

// The matrix-N multiplication operation, [out mat] = [left mat] x [right mat].
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void MulMatrixN(const T* lhs, const T* rhs, T* out) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            SCALE_TYPE buf = 0.0;
            for (size_t k = 0; k < N; ++k) {
                buf += lhs[i * N + k] * rhs[k * N + j];
            }
            out[i * N + j] = buf;
        }
    }
}

// The matrix-N multiplication operation, [out mat] = [in mat] * scale.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void MulMatrixN(const T* in, T* out, CONST_SCALE_TYPE scale) {
    MulVectorN<N * N>(in, out, scale);
}

// The matrix-N division operation, [out mat] = [left mat] / [right mat].
// This operation is not common.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void DivMatrixN(const T* lhs, const T* rhs, T* out) {
    DivVectorN<N * N>(lhs, rhs, out);
}

// The matrix-N and scale division operation,
//     [out mat] = [in mat] / scale,    if not invert.
//     [out mat] =    scale / [in mat], if invert.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void DivMatrixN(const T* in, T* out, CONST_SCALE_TYPE scale, const bool invert) {
    DivVectorN<N * N>(in, out, scale, invert);
}

// Scale the model matrix-N.
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void ScaleMatrixN(const T* vec, T* mat) {
    FOREACH_LOOP(N-1) { mat[GetMatrixNIndex<4>(i)] *= vec[i]; } FOREACH_LOOP_END;
}

// Invert operator function, only for matrix-4.
template<size_t N, typename T, typename = std::enable_if_t<N == 4 && IS_NUMBER(T)>>
constexpr void InvertMatrixN(const T* mat4, T* inv4) {
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

    SCALE_TYPE det = mat4[0] * inv4[0] +
                         mat4[1] * inv4[4] +
                         mat4[2] * inv4[8] +
                         mat4[3] * inv4[12];
    det = 1.0 / det;

    for (int i = 0; i < 16; i++) {
        inv4[i] *= det;
    }
}

// Common model matrix, only for matrix-4.
template<typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void TranslationMatrix4(const T* vec3, T* mat4) {
    FillMatrixN<4>(mat4, 0);
    FillDiagonalMatrixN<4>(mat4, 1);
    mat4[GetMatrixNIndex<4>(kAxisX, kAxisW)] = vec3[kAxisX];
    mat4[GetMatrixNIndex<4>(kAxisY, kAxisW)] = vec3[kAxisY];
    mat4[GetMatrixNIndex<4>(kAxisZ, kAxisW)] = vec3[kAxisZ];
}

template<typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr T ToRadians(const T degree) {
    return degree * (M_PI / 180.0);
}

template<typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr T ToDegree(const T radians) {
    return radians * (180.0 / M_PI);
}

// Common model matrix, only for matrix-4.
template<typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void RotationMatrix4AtAxis(T* mat4, const int axis, CONST_SCALE_TYPE degree) {
    CONST_SCALE_TYPE radians = ToRadians(degree);

    FillMatrixN<4>(mat4, 0);
    FillDiagonalMatrixN<4>(mat4, 1);

    if (axis == kAxisW) {
        return;
    }

    SCALE_TYPE vec3[3] = {0};
    vec3[axis] = 1.0;

    CONST_SCALE_TYPE rx = -vec3[kAxisX];
    CONST_SCALE_TYPE ry = -vec3[kAxisY];
    CONST_SCALE_TYPE rz = -vec3[kAxisZ];

    CONST_SCALE_TYPE cos_v = std::cos(radians);
    CONST_SCALE_TYPE sin_v = std::sin(radians);

    // x
    mat4[GetMatrixNIndex<4>(kAxisX, kAxisX)] =       1 * cos_v     + rx * rx * (1-cos_v);
    mat4[GetMatrixNIndex<4>(kAxisX, kAxisY)] = rx * ry * (1-cos_v) -      rz * sin_v;
    mat4[GetMatrixNIndex<4>(kAxisX, kAxisZ)] = rx * rz * (1-cos_v) +      ry * sin_v;

    // y
    mat4[GetMatrixNIndex<4>(kAxisY, kAxisX)] = ry * rx * (1-cos_v) +      rz * sin_v;
    mat4[GetMatrixNIndex<4>(kAxisY, kAxisY)] =       1 * cos_v     + ry * ry * (1-cos_v);
    mat4[GetMatrixNIndex<4>(kAxisY, kAxisZ)] = ry * rz * (1-cos_v) -      rx * sin_v;

    // z
    mat4[GetMatrixNIndex<4>(kAxisZ, kAxisX)] = rz * rx * (1-cos_v) -      ry * sin_v;
    mat4[GetMatrixNIndex<4>(kAxisZ, kAxisY)] = rz * ry * (1-cos_v) +      rx * sin_v;
    mat4[GetMatrixNIndex<4>(kAxisZ, kAxisZ)] =       1 * cos_v     + rz * rz * (1-cos_v);
}


// Common view matrix, only for matrix-4.
template<typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void LookatMatrix4(const T* eye,
                             const T* center,
                             const T* up,
                             T* mat4) {
    T* main_buf = (T*)std::malloc(sizeof(T) * 3 * 3);
    T* buf_x = main_buf + 0;
    T* buf_y = main_buf + 3;
    T* buf_z = main_buf + 6;

    SubVectorN<3>(eye, center, buf_z); // buf_z = eye - center
    NormalizeVectorN<3>(buf_z, buf_z); // buf_z = norm(buf_z)

    CrossProductVectorN<3>(up,    buf_z, buf_x); // buf_x = cross(up,    buf_z)
    CrossProductVectorN<3>(buf_z, buf_x, buf_y); // buf_y = cross(buf_z, buf_x)

    NormalizeVectorN<3>(buf_x, buf_x); // buf_x = norm(buf_x)
    NormalizeVectorN<3>(buf_y, buf_y); // buf_y = norm(buf_y)

    // x
    mat4[GetMatrixNIndex<4>(kAxisX, kAxisX)] = buf_x[kAxisX];
    mat4[GetMatrixNIndex<4>(kAxisX, kAxisY)] = buf_y[kAxisX];
    mat4[GetMatrixNIndex<4>(kAxisX, kAxisZ)] = buf_z[kAxisX];

    // y
    mat4[GetMatrixNIndex<4>(kAxisY, kAxisX)] = buf_x[kAxisY];
    mat4[GetMatrixNIndex<4>(kAxisY, kAxisY)] = buf_y[kAxisY];
    mat4[GetMatrixNIndex<4>(kAxisY, kAxisZ)] = buf_z[kAxisY];

    // z
    mat4[GetMatrixNIndex<4>(kAxisZ, kAxisX)] = buf_x[kAxisZ];
    mat4[GetMatrixNIndex<4>(kAxisZ, kAxisY)] = buf_y[kAxisZ];
    mat4[GetMatrixNIndex<4>(kAxisZ, kAxisZ)] = buf_z[kAxisZ];

    // all vecs
    for (int i = 0; i < 9; ++i) {
        main_buf[i] = -main_buf[i];
    }

    // w
    mat4[GetMatrixNIndex<4>(kAxisW, kAxisX)] = InnerProductVectorN<3>(buf_x, eye);
    mat4[GetMatrixNIndex<4>(kAxisW, kAxisY)] = InnerProductVectorN<3>(buf_y, eye);
    mat4[GetMatrixNIndex<4>(kAxisW, kAxisZ)] = InnerProductVectorN<3>(buf_z, eye);

    mat4[GetMatrixNIndex<4>(kAxisX, kAxisW)] = 0.0;
    mat4[GetMatrixNIndex<4>(kAxisY, kAxisW)] = 0.0;
    mat4[GetMatrixNIndex<4>(kAxisZ, kAxisW)] = 0.0;
    mat4[GetMatrixNIndex<4>(kAxisW, kAxisW)] = 1.0;

    free(main_buf);
}


// Common perspective matrix, only for matrix-4.
template<typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
constexpr void PerspectiveMatrix4(T* mat4,
                                  CONST_SCALE_TYPE fov,
                                  CONST_SCALE_TYPE aspect,
                                  CONST_SCALE_TYPE near,
                                  CONST_SCALE_TYPE far) {
    CONST_SCALE_TYPE scale_y = 1.0 / std::tan(ToRadians(fov) / 2.0);
    CONST_SCALE_TYPE scale_x = scale_y / aspect;
    CONST_SCALE_TYPE diff    = near - far;

    FillMatrixN<4>(mat4, 0.0); // clear

    mat4[GetMatrixNIndex<4>(kAxisX)] = scale_x;
    mat4[GetMatrixNIndex<4>(kAxisY)] = scale_y;
    mat4[GetMatrixNIndex<4>(kAxisZ)] = (near + far)/diff;

    mat4[GetMatrixNIndex<4>(kAxisZ, kAxisW)] = -1.0;
    mat4[GetMatrixNIndex<4>(kAxisW, kAxisZ)] = (2*far*near)/diff;
}

// basic buffer type
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
class ElementBuffer : public std::array<T, N> {};

// all data will be stored in the vector buffer
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
class VectorBuffer : public ElementBuffer<N, T> {
public:
    VectorBuffer() = default;
    VectorBuffer(T scale) { this->fill(scale); }
    VectorBuffer(std::initializer_list<T> &list) {
        const int dim = std::min(N, list.size());
        for (int i = 0; i < dim; ++i) {
            this->at(i) = *(std::begin(list) + i);
        }
    }
    VectorBuffer<N, T>& operator=(const std::initializer_list<T> &list) {
        const int dim = std::min(N, list.size());
        for (int i = 0; i < dim; ++i) {
            this->at(i) = *(std::begin(list) + i);
        }
        return *this;
    }

    static_assert(N >= 1 && N <= 4, "dim should be 1~4");
};

// basic class type for all vector-N
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
class VectorN {
public:
    static constexpr size_t kSize = N;

    virtual inline T& operator [](int idx) = 0;
    virtual inline T operator [](int idx) const = 0;
    virtual T* GetPtr() = 0;
    virtual const T* GetPtr() const = 0;

    inline std::string ToString() const {
        return VecorNToString<kSize>(GetPtr());
    }
    friend std::ostream& operator<<(std::ostream& os, const VectorN<N, T>& vec) {
        os << vec.ToString();
        return os;
    }

protected:
    constexpr void _FillVectorN(T* vec, CONST_SCALE_TYPE scale) {
        FillVectorN<kSize>(vec, scale);
    }
    constexpr void _AddVectorN(const T* lhs, const T* rhs, T* out) {
        AddVectorN<kSize>(lhs, rhs, out);
    }
    constexpr void _AddVectorN(const T* in, T* out, CONST_SCALE_TYPE scale) {
        AddVectorN<kSize>(in, out, scale);
    }
    constexpr void _SubVectorN(const T* lhs, const T* rhs, T* out) {
        SubVectorN<kSize>(lhs, rhs, out);
    }
    constexpr void _SubVectorN(const T* in, T* out, CONST_SCALE_TYPE scale, const bool invert) {
        SubVectorN<kSize>(in, out, scale, invert);
    }
    constexpr void _MulVectorN(const T* lhs, const T* rhs, T* out) {
        MulVectorN<kSize>(lhs, rhs, out);
    }
    constexpr void _MulVectorN(const T* in, T* out, CONST_SCALE_TYPE scale) {
        MulVectorN<kSize>(in, out, scale);
    }
    constexpr void _DivVectorN(const T* lhs, const T* rhs, T* out) {
        DivVectorN<kSize>(lhs, rhs, out);
    }
    constexpr void _DivVectorN(const T* in, T* out, CONST_SCALE_TYPE scale, const bool invert) {
        DivVectorN<kSize>(in, out, scale, invert);
    }
    constexpr void _CrossProductVectorN(const T* lhs, const T* rhs, T* out) {
        CrossProductVectorN<kSize>(lhs, rhs, out);
    }
    constexpr T _InnerProductVectorN(const T* lhs, const T* rhs) {
        return InnerProductVectorN<kSize>(lhs, rhs);
    }
    constexpr void _NormalizeVectorN(const T* in, T* out) {
        NormalizeVectorN<kSize>(in, out);
    }
};

#define VECTOR_N_FUNCTIONS \
    virtual inline T& operator [](int idx) override { \
        return data[idx]; \
    } \
    virtual inline T operator [](int idx) const override { \
        return data[idx]; \
    } \
    virtual inline T* GetPtr() override { \
        return data.data(); \
    } \
    virtual inline const T* GetPtr() const override { \
        return data.data(); \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_VectorType, V)>> \
    _VectorType operator+(V&& rhs) { \
        _VectorType out; \
        this->_AddVectorN(GetPtr(), rhs.GetPtr(), out.GetPtr()); \
        return out; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _VectorType operator+(S scale) { \
        _VectorType out; \
        this->_AddVectorN(GetPtr(), out.GetPtr(), scale); \
        return out; \
    } \
    template<typename S, typename V, typename = std::enable_if_t<IS_NUMBER(S) && IS_SAME(_VectorType, V)>> \
    friend _VectorType operator+(S scale, V&& rhs) { \
        _VectorType out; \
        rhs._AddVectorN(rhs.GetPtr(), out.GetPtr(), scale); \
        return out; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_VectorType, V)>> \
    _VectorType& operator+=(V&& rhs) { \
        this->_AddVectorN(GetPtr(), rhs.GetPtr(), GetPtr()); \
        return *this; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _VectorType& operator+=(S scale) { \
        this->_AddVectorN(GetPtr(), GetPtr(), scale); \
        return *this; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_VectorType, V)>> \
    _VectorType operator-(V&& rhs) { \
        _VectorType out; \
        this->_SubVectorN(GetPtr(), rhs.GetPtr(), out.GetPtr()); \
        return out; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _VectorType operator-(S scale) { \
        _VectorType out; \
        this->_SubVectorN(GetPtr(), out.GetPtr(), scale, false); \
        return out; \
    } \
    template<typename S, typename V, typename = std::enable_if_t<IS_NUMBER(S) && IS_SAME(_VectorType, V)>> \
    friend _VectorType operator-(S scale, V&& rhs) { \
        _VectorType out; \
        rhs._SubVectorN(rhs.GetPtr(), out.GetPtr(), scale, true); \
        return out; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_VectorType, V)>> \
    _VectorType& operator-=(V&& rhs) { \
        this->_SubVectorN(GetPtr(), rhs.GetPtr(), GetPtr()); \
        return *this; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _VectorType& operator-=(S scale) { \
        this->_SubVectorN(GetPtr(), GetPtr(), scale, false); \
        return *this; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_VectorType, V)>> \
    _VectorType operator*(V&& rhs) { \
        _VectorType out; \
        this->_MulVectorN(GetPtr(), rhs.GetPtr(), out.GetPtr()); \
        return out; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _VectorType operator*(S scale) { \
        _VectorType out; \
        this->_MulVectorN(GetPtr(), out.GetPtr(), scale); \
        return out; \
    } \
    template<typename S, typename V, typename = std::enable_if_t<IS_NUMBER(S) && IS_SAME(_VectorType, V)>> \
    friend _VectorType operator*(S scale, V&& rhs) { \
        _VectorType out; \
        rhs._MulVectorN(rhs.GetPtr(), out.GetPtr(), scale); \
        return out; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_VectorType, V)>> \
    _VectorType& operator*=(V&& rhs) { \
        this->_MulVectorN(GetPtr(), rhs.GetPtr(), GetPtr()); \
        return *this; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _VectorType& operator*=(S scale) { \
        this->_MulVectorN(GetPtr(), GetPtr(), scale); \
        return *this; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_VectorType, V)>> \
    _VectorType operator/(V&& rhs) { \
        _VectorType out; \
        this->_DivVectorN(GetPtr(), rhs.GetPtr(), out.GetPtr()); \
        return out; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _VectorType operator/(S scale) { \
        _VectorType out; \
        this->_DivVectorN(GetPtr(), out.GetPtr(), scale, false); \
        return out; \
    } \
    template<typename S, typename V, typename = std::enable_if_t<IS_NUMBER(S) && IS_SAME(_VectorType, V)>> \
    friend _VectorType operator/(S scale, V&& rhs) { \
        _VectorType out; \
        rhs._DivVectorN(rhs.GetPtr(), out.GetPtr(), scale, true); \
        return out; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_VectorType, V)>> \
    _VectorType& operator/=(V&& rhs) { \
        this->_DivVectorN(GetPtr(), rhs.GetPtr(), GetPtr()); \
        return *this; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _VectorType& operator/=(S scale) { \
        this->_DivVectorN(GetPtr(), GetPtr(), scale, false); \
        return *this; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_VectorType, V)>> \
    constexpr _VectorType CrossProduct(V&& rhs) { \
        _VectorType out; \
        this->_CrossProductVectorN(GetPtr(), rhs.GetPtr(), out.GetPtr()); \
        return out; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_VectorType, V)>> \
    constexpr T InnerProduct(V&& rhs) { \
        return this->_InnerProductVectorN(GetPtr(), rhs.GetPtr()); \
    } \
    constexpr _VectorType& Normalize() { \
        this->_NormalizeVectorN(GetPtr(), GetPtr()); \
        return *this; \
    }

// implement vector-3
template<typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
class Vector3 : public VectorN<3, T> {
public:
    using _VectorN = VectorN<3, T>;
    using _VectorType = Vector3<T>;
    using _VectorBuffer = VectorBuffer<_VectorN::kSize, T>;

    constexpr Vector3() : x(0), y(0), z(0) {}
    constexpr Vector3(T scale) : x(scale), y(scale), z(scale) {}
    constexpr Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
    constexpr Vector3(std::initializer_list<T> &list) : data(list) {}
    constexpr Vector3(std::initializer_list<T> &&list) : data(list) {}

    VECTOR_N_FUNCTIONS;

    union {
        struct { T x, y, z; };
        _VectorBuffer data;
    };
};

// implement vector-4
template<typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
class Vector4 : public VectorN<4, T> {
public:
    using _VectorN = VectorN<4, T>;
    using _VectorType = Vector4<T>;
    using _VectorBuffer = VectorBuffer<_VectorN::kSize, T>;

    constexpr Vector4() : x(0), y(0), z(0), w(0) {}
    constexpr Vector4(T scale) : x(scale), y(scale), z(scale), w(scale) {}
    constexpr Vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
    constexpr Vector4(std::initializer_list<T> &list) : data(list) {}
    constexpr Vector4(std::initializer_list<T> &&list) : data(list) {}

    VECTOR_N_FUNCTIONS;

    union {
        struct { T x, y, z, w; };
        _VectorBuffer data;
    };
};

// matrix class type
template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
class MatrixBuffer : public ElementBuffer<N * N, T> {
public:
    MatrixBuffer() = default;
    MatrixBuffer(T scale) { this->fill(scale); }
    MatrixBuffer(std::initializer_list<T> &list) {
        const int dim2 = std::min(N * N, list.size());
        for (int i = 0; i < dim2; ++i) {
            this->at(i) = *(std::begin(list) + i);
        }
    }

    static_assert(N >= 1 && N <= 4, "dim should be 1~4");
};

template<size_t N, typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
class MatrixN {
public:
    static constexpr size_t kSize = N;

    virtual inline VectorBuffer<N, T>& operator [](int idx) = 0;
    virtual inline VectorBuffer<N, T> operator [](int idx) const = 0;
    virtual T* GetPtr() = 0;
    virtual const T* GetPtr() const = 0;

    inline std::string ToString() const {
        return MatrixNToString<kSize>(GetPtr());
    }
    friend std::ostream& operator<<(std::ostream& os, const MatrixN<N, T>& mat) {
        os << mat.ToString();
        return os;
    }

protected:
    constexpr void _FillMatrixN(T* vec, CONST_SCALE_TYPE scale) {
        FillMatrixN<kSize>(vec, scale);
    }
    constexpr void _FillDiagonalMatrixN(T* vec, CONST_SCALE_TYPE scale) {
        FillDiagonalMatrixN<kSize>(vec, scale);
    }
    constexpr void _AddMatrixN(const T* lhs, const T* rhs, T* out) {
        AddMatrixN<kSize>(lhs, rhs, out);
    }
    constexpr void _AddMatrixN(const T* in, T* out, CONST_SCALE_TYPE scale) {
        AddMatrixN<kSize>(in, out, scale);
    }
    constexpr void _SubMatrixN(const T* lhs, const T* rhs, T* out) {
        SubMatrixN<kSize>(lhs, rhs, out);
    }
    constexpr void _SubMatrixN(const T* in, T* out, CONST_SCALE_TYPE scale, const bool invert) {
        SubMatrixN<kSize>(in, out, scale, invert);
    }
    constexpr void _MulMatrixN(const T* lhs, const T* rhs, T* out) {
        MulMatrixN<kSize>(lhs, rhs, out);
    }
    constexpr void _MulMatrixN(const T* in, T* out, CONST_SCALE_TYPE scale) {
        MulMatrixN<kSize>(in, out, scale);
    }
    constexpr void _DivMatrixN(const T* lhs, const T* rhs, T* out) {
        DivMatrixN<kSize>(lhs, rhs, out);
    }
    constexpr void _DivMatrixN(const T* in, T* out, CONST_SCALE_TYPE scale, const bool invert) {
        DivMatrixN<kSize>(in, out, scale, invert);
    }
    constexpr void _InvertMatrixN(const T* in, T* out) {
        InvertMatrixN<kSize>(in, out);
    }
    constexpr void _ScaleMatrixN(const T* vec, T* mat) {
        ScaleMatrixN<kSize>(vec, mat);
    }
};

#define MATRIX_N_FUNCTIONS \
    virtual inline _VectorBuffer& operator [](int idx) override { \
        return mdata[idx]; \
    } \
    virtual inline _VectorBuffer operator [](int idx) const override { \
        return mdata[idx]; \
    } \
    virtual inline T* GetPtr() override { \
        return ldata.data(); \
    } \
    virtual inline const T* GetPtr() const override { \
        return ldata.data(); \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_MatrixType, V)>> \
    _MatrixType operator+(V&& rhs) { \
        _MatrixType out; \
        this->_AddMatrixN(GetPtr(), rhs.GetPtr(), out.GetPtr()); \
        return out; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _MatrixType operator+(S scale) { \
        _MatrixType out; \
        this->_AddMatrixN(GetPtr(), out.GetPtr(), scale); \
        return out; \
    } \
    template<typename S, typename V, typename = std::enable_if_t<IS_NUMBER(S) && IS_SAME(_MatrixType, V)>> \
    friend _MatrixType operator+(S scale, V&& rhs) { \
        _MatrixType out; \
        rhs._AddMatrixN(rhs.GetPtr(), out.GetPtr(), scale); \
        return out; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_MatrixType, V)>> \
    _MatrixType& operator+=(V&& rhs) { \
        this->_AddMatrixN(GetPtr(), rhs.GetPtr(), GetPtr()); \
        return *this; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _MatrixType& operator+=(S scale) { \
        this->_AddMatrixN(GetPtr(), GetPtr(), scale); \
        return *this; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_MatrixType, V)>> \
    _MatrixType operator-(V&& rhs) { \
        _MatrixType out; \
        this->_SubMatrixN(GetPtr(), rhs.GetPtr(), out.GetPtr()); \
        return out; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _MatrixType operator-(S scale) { \
        _MatrixType out; \
        this->_SubMatrixN(GetPtr(), out.GetPtr(), scale, false); \
        return out; \
    } \
    template<typename S, typename V, typename = std::enable_if_t<IS_NUMBER(S) && IS_SAME(_MatrixType, V)>> \
    friend _MatrixType operator-(S scale, V&& rhs) { \
        _MatrixType out; \
        rhs._SubMatrixN(rhs.GetPtr(), out.GetPtr(), scale, true); \
        return out; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_MatrixType, V)>> \
    _MatrixType& operator-=(V&& rhs) { \
        this->_SubMatrixN(GetPtr(), rhs.GetPtr(), GetPtr()); \
        return *this; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _MatrixType& operator-=(S scale) { \
        this->_SubMatrixN(GetPtr(), GetPtr(), scale, false); \
        return *this; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_MatrixType, V)>> \
    _MatrixType operator*(V&& rhs) { \
        _MatrixType out; \
        this->_MulMatrixN(GetPtr(), rhs.GetPtr(), out.GetPtr()); \
        return out; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _MatrixType operator*(S scale) { \
        _MatrixType out; \
        this->_MulMatrixN(GetPtr(), out.GetPtr(), scale); \
        return out; \
    } \
    template<typename S, typename V, typename = std::enable_if_t<IS_NUMBER(S) && IS_SAME(_MatrixType, V)>> \
    friend _MatrixType operator*(S scale, V&& rhs) { \
        _MatrixType out; \
        rhs._MulMatrixN(rhs.GetPtr(), out.GetPtr(), scale); \
        return out; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_MatrixType, V)>> \
    _MatrixType& operator*=(V&& rhs) { \
        this->_MulMatrixN(GetPtr(), rhs.GetPtr(), GetPtr()); \
        return *this; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _MatrixType& operator*=(S scale) { \
        this->_MulMatrixN(GetPtr(), GetPtr(), scale); \
        return *this; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_MatrixType, V)>> \
    _MatrixType operator/(V&& rhs) { \
        _MatrixType out; \
        this->_DivMatrixN(GetPtr(), rhs.GetPtr(), out.GetPtr()); \
        return out; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _MatrixType operator/(S scale) { \
        _MatrixType out; \
        this->_DivMatrixN(GetPtr(), out.GetPtr(), scale, false); \
        return out; \
    } \
    template<typename S, typename V, typename = std::enable_if_t<IS_NUMBER(S) && IS_SAME(_MatrixType, V)>> \
    friend _MatrixType operator/(S scale, V&& rhs) { \
        _MatrixType out; \
        rhs._DivMatrixN(rhs.GetPtr(), out.GetPtr(), scale, true); \
        return out; \
    } \
    template<typename V, typename = std::enable_if_t<IS_SAME(_MatrixType, V)>> \
    _MatrixType& operator/=(V&& rhs) { \
        this->_DivMatrixN(GetPtr(), rhs.GetPtr(), GetPtr()); \
        return *this; \
    } \
    template<typename S, typename = std::enable_if_t<IS_NUMBER(S)>> \
    _MatrixType& operator/=(S scale) { \
        this->_DivMatrixN(GetPtr(), GetPtr(), scale, false); \
        return *this; \
    } \
    constexpr _MatrixType& Invert() { \
        _MatrixType buf = *this; \
        this->_InvertMatrixN(buf.GetPtr(), GetPtr()); \
        return *this; \
    } \
    template<typename V, typename = std::enable_if_t<IS_BASE_OF(_OpVectorN, V)>> \
    constexpr _MatrixType& Scale(V&& vec) { \
        this->_ScaleMatrixN(vec.GetPtr(), GetPtr()); \
        return *this; \
    }

template<typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
class Matrix3 : public MatrixN<3, T> {
public:
    using _MatrixN = MatrixN<3, T>;
    using _MatrixType = Matrix3<T>;
    using _MatrixBuffer = MatrixBuffer<_MatrixN::kSize, T>;
    using _VectorBuffer = VectorBuffer<_MatrixN::kSize, T>;
    using _OpVectorN = VectorN<_MatrixN::kSize-1, T>;

    constexpr Matrix3() : ldata(0) {}
    constexpr Matrix3(T scale) : ldata(0) { this->_FillDiagonalMatrixN(GetPtr(), scale); }
    constexpr Matrix3(std::initializer_list<T>& list) : ldata(list) {}
    constexpr Matrix3(std::initializer_list<T>&& list) : ldata(list) {}

    MATRIX_N_FUNCTIONS;

    union {
        std::array<_VectorBuffer, _MatrixN::kSize> mdata;
        _MatrixBuffer ldata;
    };
};

template<typename T, typename = std::enable_if_t<IS_NUMBER(T)>>
class Matrix4 : public MatrixN<4, T> {
public:
    using _MatrixN = MatrixN<4, T>;
    using _MatrixType = Matrix4<T>;
    using _MatrixBuffer = MatrixBuffer<_MatrixN::kSize, T>;
    using _VectorBuffer = VectorBuffer<_MatrixN::kSize, T>;
    using _OpVectorN = VectorN<_MatrixN::kSize-1, T>;

    constexpr Matrix4() : ldata(0) {}
    constexpr Matrix4(T scale) : ldata(0) { this->_FillDiagonalMatrixN(GetPtr(), scale); }
    constexpr Matrix4(std::initializer_list<T>& list) : ldata(list) {}
    constexpr Matrix4(std::initializer_list<T>&& list) : ldata(list) {}

    MATRIX_N_FUNCTIONS;

    union {
        std::array<_VectorBuffer, _MatrixN::kSize> mdata;
        _MatrixBuffer ldata;
    };

    template<typename V, typename = std::enable_if_t<IS_SAME(Vector3<T>, V)>>
    static constexpr _MatrixType GetTranslation(V&& vec) {
        _MatrixType out;
        TranslationMatrix4(vec.GetPtr(), out.GetPtr());
        return out;
    }
    static constexpr _MatrixType GetRotation(const int axis,
                                             CONST_SCALE_TYPE degree) {
        _MatrixType out;
        RotationMatrix4AtAxis(out.GetPtr(), axis, degree);
        return out;
    }
    template<
        typename V1, typename V2, typename V3,
        typename = std::enable_if_t<
            IS_SAME(Vector3<T>, V1) && IS_SAME(Vector3<T>, V2) && IS_SAME(Vector3<T>, V3)
        >
    >
    static constexpr _MatrixType GetLookat(V1&& eye, V2&& center, V3&& up) {
        _MatrixType out;
        LookatMatrix4(eye.GetPtr(), center.GetPtr(),
                      up.GetPtr(), out.GetPtr());
        return out;
    }
    static constexpr _MatrixType GetPerspective(CONST_SCALE_TYPE fov,
                                                CONST_SCALE_TYPE aspect,
                                                CONST_SCALE_TYPE near,
                                                CONST_SCALE_TYPE far) {
        _MatrixType out;
        PerspectiveMatrix4(out.GetPtr(), fov, aspect, near, far);
        return out;
    }
};

using Vector3i = Vector3<int>;
using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;
using Vector4i = Vector4<int>;
using Vector4f = Vector4<float>;
using Vector4d = Vector4<double>;
using Matrix3i = Matrix3<int>;
using Matrix3f = Matrix3<float>;
using Matrix3d = Matrix3<double>;
using Matrix4i = Matrix4<int>;
using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;

#undef FOREACH_LOOP
#undef FOREACH_LOOP_END
#undef CONST_SCALE_TYPE
#undef SCALE_TYPE
#undef IS_NUMBER
#undef IS_SAME
#undef IS_BASE_OF
#undef VECTOR_N_FUNCTIONS
#undef MATRIX_N_FUNCTIONS
