#ifndef GRAPHICS_MATH_H_INCLUDE
#define GRAPHICS_MATH_H_INCLUDE

#define _USE_MATH_DEFINES // for math.h

#include <iostream>
#include <iomanip>

#include <array>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <initializer_list>

#define CHECK_REF_TYPE(V)                                      \
{                                                              \
    if (std::is_lvalue_reference<decltype(V)>::value) {        \
        printf("lval ref\n");                                  \
    } else if (std::is_lvalue_reference<decltype(V)>::value) { \
        printf("rval ref\n");                                  \
    } else {                                                   \
        printf("not ref\n");                                   \
    }                                                          \
}

#define STATIC_ASSERT_TYPE(T)                                  \
static_assert(std::is_same<T, float>::value ||                 \
                  std::is_same<T, double>::value,              \
                  "Only support for the float and double type.");

#define VEC3_DIM_SIZE (3)
#define MAT4_DIM_SIZE (4)

#define CONST_SCALE_TYPE const double
#define SCALE_TYPE double

static constexpr int kAxisX = 0;
static constexpr int kAxisZ = 1;
static constexpr int kAxisY = 2;
static constexpr int kAxisW = 3;


// C-like functions.

template<typename T>
constexpr T GetX(const T* vec) {
    STATIC_ASSERT_TYPE(T);
    return vec[kAxisX];
}
template<typename T>
constexpr T GetY(const T* vec) {
    STATIC_ASSERT_TYPE(T);
    return vec[kAxisY];
}

template<typename T>
constexpr T GetZ(const T* vec) {
    STATIC_ASSERT_TYPE(T);
    return vec[kAxisZ];
}

template<typename T>
constexpr T GetW(const T* vec) {
    STATIC_ASSERT_TYPE(T);
    return vec[kAxisW];
}

constexpr int GetMat4Index(const int y, const int x) {
    return y * MAT4_DIM_SIZE + x;
}

constexpr int GetMat4Index(const int dim) {
    return GetMat4Index(dim, dim);
}

template<typename T>
constexpr T ToRadians(const T degree) {
    STATIC_ASSERT_TYPE(T);
    return degree * (M_PI / 180.f);
}

template<typename T>
constexpr T ToDegree(const T radians) {
    STATIC_ASSERT_TYPE(T);
    return radians * (180.f / M_PI);
}

template<typename T>
std::string FloatToString(T val) {
    STATIC_ASSERT_TYPE(T);
    auto out = std::ostringstream{};
    int p = 6;
    int w = p + 6;
    out << std::fixed
            << std::setw(w)
            << std::setprecision(p)
            << val;
    return out.str();
};


// Convert the vector3 to std::string.
template<typename T>
inline std::string Vec3ToString(const T* vec3) {
    STATIC_ASSERT_TYPE(T);

    auto out = std::ostringstream{};
    out
        << '('
        << FloatToString(GetX(vec3)) << ", "
        << FloatToString(GetY(vec3)) << ", "
        << FloatToString(GetZ(vec3)) << ")";
    return out.str();
}

// Convert the vector4 to std::string.
template<typename T>
inline std::string Vec4ToString(const T* vec4) {
    STATIC_ASSERT_TYPE(T);

    auto out = std::ostringstream{};
    out
        << '('
        << FloatToString(GetX(vec4)) << ", "
        << FloatToString(GetY(vec4)) << ", "
        << FloatToString(GetZ(vec4)) << ", "
        << FloatToString(GetW(vec4)) << ")";
    return out.str();
}

// Convert the matrix4 to std::string.
template<typename T>
inline std::string Mat4ToString(const T* mat4) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = MAT4_DIM_SIZE;

    auto out = std::ostringstream{};
    out << "{\n";
    for (int i = 0; i < N; ++i) {
        const T* offset = mat4 + i * N;
        out
            << "  ("
            << FloatToString(offset[kAxisX]) << ", "
            << FloatToString(offset[kAxisY]) << ", "
            << FloatToString(offset[kAxisZ]) << ", "
            << FloatToString(offset[kAxisW]) << ")";
        if (i != N-1) {
            out << "\n";
        }
    }
    out << "\n}";
    return out.str();
}


// Print the vector3 elements.
template<typename T>
inline void PrintVec3(const T* vec3) {
    STATIC_ASSERT_TYPE(T);

    std::cout << Vec3ToString(vec3) << std::endl;
}

// Print the vector4 elements.
template<typename T>
inline void PrintVec4(const T* vec4) {
    STATIC_ASSERT_TYPE(T);

    std::cout << Vec4ToString(vec4) << std::endl;
}

// Print the matrix4 elements.
template<typename T>
inline void PrintMat4(const T* mat4) {
    STATIC_ASSERT_TYPE(T);

    std::cout << Mat4ToString(mat4) << std::endl;
}

// Fill the vector3.
template<typename T>
inline void FillVec3(T* vec3, CONST_SCALE_TYPE scale) {
    STATIC_ASSERT_TYPE(T);

    vec3[kAxisX] =
        vec3[kAxisY] =
        vec3[kAxisZ] =
        scale;
}

// The vector3 addition operation, [out vec] = [left vec] + [right vec].
template<typename T>
inline void AddVec3(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);

    out[kAxisX] = lhs[kAxisX] + rhs[kAxisX];
    out[kAxisY] = lhs[kAxisY] + rhs[kAxisY];
    out[kAxisZ] = lhs[kAxisZ] + rhs[kAxisZ];
}

// The vector3 and scale addition operation, [out vec] = [in vec] + scale.
template<typename T>
inline void AddVec3(const T* in, T* out, CONST_SCALE_TYPE scale) {
    STATIC_ASSERT_TYPE(T);

    out[kAxisX] = in[kAxisX] + scale;
    out[kAxisY] = in[kAxisY] + scale;
    out[kAxisZ] = in[kAxisZ] + scale;
}

// The vector3 subtraction operation, [out vec] = [left vec] - [right vec].
template<typename T>
inline void SubVec3(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);

    out[kAxisX] = lhs[kAxisX] - rhs[kAxisX];
    out[kAxisY] = lhs[kAxisY] - rhs[kAxisY];
    out[kAxisZ] = lhs[kAxisZ] - rhs[kAxisZ];
}

// The vector3 and scale subtraction operation, 
//     [out vec] = [in vec] - scale,    if not invert.
//     [out vec] =    scale - [in vec], if invert.
template<typename T>
inline void SubVec3(const T* in, T* out, CONST_SCALE_TYPE scale, bool invert) {
    STATIC_ASSERT_TYPE(T);

    if (!invert) {
        out[kAxisX] = in[kAxisX] - scale;
        out[kAxisY] = in[kAxisY] - scale;
        out[kAxisZ] = in[kAxisZ] - scale;
    } else {
        out[kAxisX] = scale - in[kAxisX];
        out[kAxisY] = scale - in[kAxisY];
        out[kAxisZ] = scale - in[kAxisZ];
    }
}

// The vector3 multiplication operation, [out vec] = [left vec] x [right vec].
template<typename T>
inline void MulVec3(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);

    out[kAxisX] = lhs[kAxisX] * rhs[kAxisX];
    out[kAxisY] = lhs[kAxisY] * rhs[kAxisY];
    out[kAxisZ] = lhs[kAxisZ] * rhs[kAxisZ];
}

// The vector3 multiplication operation, [out vec] = [in vec] x scale.
template<typename T>
inline void MulVec3(const T* in, T* out, CONST_SCALE_TYPE scale) {
    STATIC_ASSERT_TYPE(T);

    out[kAxisX] = in[kAxisX] * scale;
    out[kAxisY] = in[kAxisY] * scale;
    out[kAxisZ] = in[kAxisZ] * scale;
}

// The vector3 division operation, [out vec] = [left vec] / [right vec].
// This operation is not common.
template<typename T>
inline void DivVec3(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);

    out[kAxisX] = lhs[kAxisX] / rhs[kAxisX];
    out[kAxisY] = lhs[kAxisY] / rhs[kAxisY];
    out[kAxisZ] = lhs[kAxisZ] / rhs[kAxisZ];
}

// The vector3 and scale division operation, 
//     [out vec] = [in vec] / scale,    if not invert.
//     [out vec] =    scale / [in vec], if invert.
template<typename T>
inline void DivVec3(const T* in, T* out, CONST_SCALE_TYPE scale, bool invert) {
    STATIC_ASSERT_TYPE(T);

    if (!invert) {
        out[kAxisX] = in[kAxisX] / scale;
        out[kAxisY] = in[kAxisY] / scale;
        out[kAxisZ] = in[kAxisZ] / scale;
    } else {
        out[kAxisX] = scale / in[kAxisX];
        out[kAxisY] = scale / in[kAxisY];
        out[kAxisZ] = scale / in[kAxisZ];
    }
}

// The vector3 cross product operation.
// wiki: https://en.wikipedia.org/wiki/Cross_product
template<typename T>
inline void CrossproductVec3(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);

    out[kAxisX] = lhs[kAxisY] * rhs[kAxisZ] - lhs[kAxisZ] * rhs[kAxisY];
    out[kAxisY] = lhs[kAxisZ] * rhs[kAxisX] - lhs[kAxisX] * rhs[kAxisZ];
    out[kAxisZ] = lhs[kAxisX] * rhs[kAxisY] - lhs[kAxisY] * rhs[kAxisX];
}

// The vector3 inner product operation.
// wiki: https://en.wikipedia.org/wiki/Inner_product_space
template<typename T>
inline T InnerproductVec3(const T* lhs, const T* rhs) {
    STATIC_ASSERT_TYPE(T);

    SCALE_TYPE val = 0.f;
    val += lhs[kAxisX] * rhs[kAxisX];
    val += lhs[kAxisY] * rhs[kAxisY];
    val += lhs[kAxisZ] * rhs[kAxisZ];
    return val;
}

// The vector3 normalizing operation. It should be
// equal to
//     |a| = sqrt(inner product(a, a))
//     out = a / |a|
template<typename T>
inline void NormalizingVec3(const T* in, T* out) {
    STATIC_ASSERT_TYPE(T);

    SCALE_TYPE scale = 1.f/std::sqrt(InnerproductVec3(in, in));
    out[kAxisX] *= scale;
    out[kAxisY] *= scale;
    out[kAxisZ] *= scale;
}

// The matrix4 addition operation, [out mat] = [left mat] + [right mat].
template<typename T>
inline void AddMat4(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = MAT4_DIM_SIZE;

    for (int i = 0; i < N; ++i) {
        const int xx = GetMat4Index(i, kAxisX);
        const int yy = GetMat4Index(i, kAxisY);
        const int zz = GetMat4Index(i, kAxisZ);
        const int ww = GetMat4Index(i, kAxisW);

        out[xx] = lhs[xx] + rhs[xx];
        out[yy] = lhs[yy] + rhs[yy];
        out[zz] = lhs[zz] + rhs[zz];
        out[ww] = lhs[ww] + rhs[ww];
    }
}

// The matrix4 and scale addition operation, [out mat] = [in mat] + scale.
template<typename T>
inline void AddMat4(const T* in, T* out, CONST_SCALE_TYPE scale) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = MAT4_DIM_SIZE;

    for (int i = 0; i < N; ++i) {
        const int xx = GetMat4Index(i, kAxisX);
        const int yy = GetMat4Index(i, kAxisY);
        const int zz = GetMat4Index(i, kAxisZ);
        const int ww = GetMat4Index(i, kAxisW);

        out[xx] = in[xx] + scale;
        out[yy] = in[yy] + scale;
        out[zz] = in[zz] + scale;
        out[ww] = in[ww] + scale;
    }
}

// The matrix4 subtraction operation, [out mat] = [left mat] - [right mat].
template<typename T>
inline void SubMat4(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = MAT4_DIM_SIZE;

    for (int i = 0; i < N; ++i) {
        const int xx = GetMat4Index(i, kAxisX);
        const int yy = GetMat4Index(i, kAxisY);
        const int zz = GetMat4Index(i, kAxisZ);
        const int ww = GetMat4Index(i, kAxisW);

        out[xx] = lhs[xx] - rhs[xx];
        out[yy] = lhs[yy] - rhs[yy];
        out[zz] = lhs[zz] - rhs[zz];
        out[ww] = lhs[ww] - rhs[ww];
    }
}

// The matrix4 and scale subtraction operation, 
//     [out vec] = [in vec] - scale,    if not invert.
//     [out vec] =    scale - [in vec], if invert.
template<typename T>
inline void SubMat4(const T* in, T* out, CONST_SCALE_TYPE scale, bool invert) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = MAT4_DIM_SIZE;

    for (int i = 0; i < N; ++i) {
        const int xx = GetMat4Index(i, kAxisX);
        const int yy = GetMat4Index(i, kAxisY);
        const int zz = GetMat4Index(i, kAxisZ);
        const int ww = GetMat4Index(i, kAxisW);

        if (!invert) {
            out[xx] = in[xx] - scale;
            out[yy] = in[yy] - scale;
            out[zz] = in[zz] - scale;
            out[ww] = in[ww] - scale;
        } else {
            out[xx] = scale - in[xx];
            out[yy] = scale - in[yy];
            out[zz] = scale - in[zz];
            out[ww] = scale - in[ww];
        }
    }
}

// The matrix4 multiplication operation
// wiki: https://en.wikipedia.org/wiki/Matrix_multiplication
template<typename T>
inline void MulMat4(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = MAT4_DIM_SIZE;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            CONST_SCALE_TYPE part0 = lhs[i * N + 0] * rhs[0 * N + j];
            CONST_SCALE_TYPE part1 = lhs[i * N + 1] * rhs[1 * N + j];
            CONST_SCALE_TYPE part2 = lhs[i * N + 2] * rhs[2 * N + j];
            CONST_SCALE_TYPE part3 = lhs[i * N + 3] * rhs[3 * N + j];

            out[i * N + j] = part0 + part1 + part2 + part3;
        }
    }
}

// The matrix4 multiplication operation, [out vec] = [in vec] x scale.
template<typename T>
inline void MulMat4(const T* in, T* out, CONST_SCALE_TYPE scale) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = MAT4_DIM_SIZE;

    for (int i = 0; i < N; ++i) {
        const int xx = GetMat4Index(i, kAxisX);
        const int yy = GetMat4Index(i, kAxisY);
        const int zz = GetMat4Index(i, kAxisZ);
        const int ww = GetMat4Index(i, kAxisW);

        out[xx] = in[xx] * scale;
        out[yy] = in[yy] * scale;
        out[zz] = in[zz] * scale;
        out[ww] = in[ww] * scale;
    }
}

// The matrix4 and vector3 multiplication operation, [out vec3] = [mat4] x [vec3].
template<typename T>
inline void MulMat4AndVec3(const T* mat4, const T* vec3, T* out) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = VEC3_DIM_SIZE;

    for (int i = 0; i < N; ++i) {
        const int xx = GetMat4Index(i, kAxisX);
        const int yy = GetMat4Index(i, kAxisY);
        const int zz = GetMat4Index(i, kAxisZ);
        const int ww = GetMat4Index(i, kAxisW);

        SCALE_TYPE temp = 0.f; 
        temp += mat4[xx] * vec3[kAxisX];
        temp += mat4[yy] * vec3[kAxisY];
        temp += mat4[zz] * vec3[kAxisZ];
        temp += mat4[ww] * 1.f;

        out[i] = temp;
    }
}

// The matrix4 and vector3 multiplication operation, [out vec4] = [mat4] x [vec4].
template<typename T>
inline void MulMat4AndVec4(const T* mat4, const T* vec4, T* out) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = MAT4_DIM_SIZE;

    for (int i = 0; i < N; ++i) {
        const int xx = GetMat4Index(i, kAxisX);
        const int yy = GetMat4Index(i, kAxisY);
        const int zz = GetMat4Index(i, kAxisZ);
        const int ww = GetMat4Index(i, kAxisW);

        SCALE_TYPE temp = 0.f; 
        temp += mat4[xx] * vec4[kAxisX];
        temp += mat4[yy] * vec4[kAxisY];
        temp += mat4[zz] * vec4[kAxisZ];
        temp += mat4[ww] * vec4[kAxisW];

        out[i] = temp;
    }
}

// The matrix4 and scale division operation, 
//     [out vec] = [in vec] / scale,    if not invert.
//     [out vec] =    scale / [in vec], if invert.
template<typename T>
inline void DivMat4(const T* in, T* out, CONST_SCALE_TYPE scale, bool invert) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = MAT4_DIM_SIZE;

    for (int i = 0; i < N; ++i) {
        const int xx = GetMat4Index(i, kAxisX);
        const int yy = GetMat4Index(i, kAxisY);
        const int zz = GetMat4Index(i, kAxisZ);
        const int ww = GetMat4Index(i, kAxisW);

        if (!invert) {
            out[xx] = in[xx] / scale;
            out[yy] = in[yy] / scale;
            out[zz] = in[zz] / scale;
            out[ww] = in[ww] / scale;
        } else {
            out[xx] = scale / in[xx];
            out[yy] = scale / in[yy];
            out[zz] = scale / in[zz];
            out[ww] = scale / in[ww];
        }
    }
}

// Fill the all elements for matrix4.
template<typename T>
inline void FillMat4(T* mat4, CONST_SCALE_TYPE scale) {
    STATIC_ASSERT_TYPE(T);

    mat4[0] =  mat4[1] =  mat4[2] =  mat4[3] =
    mat4[4] =  mat4[5] =  mat4[6] =  mat4[7] =
    mat4[8] =  mat4[9] =  mat4[10] = mat4[11] =
    mat4[12] = mat4[13] = mat4[14] = mat4[15] = scale;
}

// Fill the diagonal elements for matrix4.
template<typename T>
inline void FillDiagonalMat4(T* mat4, CONST_SCALE_TYPE scale, bool clear) {
    STATIC_ASSERT_TYPE(T);

    if (clear) {
        FillMat4(mat4, 0.0f); // clear
    }

    mat4[GetMat4Index(kAxisX)] = scale;
    mat4[GetMat4Index(kAxisY)] = scale;
    mat4[GetMat4Index(kAxisZ)] = scale;
    mat4[GetMat4Index(kAxisW)] = scale;
}

// Fill the identity matrix4.
template<typename T>
inline void FillIdentityMat4(T* mat4, bool clear) {
    STATIC_ASSERT_TYPE(T);

    FillDiagonalMat4(mat4, 1.0f, clear);
}

// Fast compute the inverse of the 4x4 matrix.
template<typename T>
inline void InvertMat4(const T* mat4, T* inv4) {
    STATIC_ASSERT_TYPE(T);

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
    det = 1.0f / det;

    for (int i = 0; i < 16; i++) {
        inv4[i] *= det;
    }
}

template<typename T>
inline void TranslationMat4(const T *vec3, T* mat4) {
    STATIC_ASSERT_TYPE(T);

    FillIdentityMat4(mat4, true);

    mat4[GetMat4Index(kAxisX, kAxisW)] = vec3[kAxisX];
    mat4[GetMat4Index(kAxisY, kAxisW)] = vec3[kAxisY];
    mat4[GetMat4Index(kAxisZ, kAxisW)] = vec3[kAxisZ];
}

template<typename T>
inline void RotationMat4AtAxis(T* mat4, const int axis, CONST_SCALE_TYPE degree) {
    STATIC_ASSERT_TYPE(T);
    CONST_SCALE_TYPE radians = ToRadians(degree);

    FillIdentityMat4(mat4, true);

    if (axis == kAxisW) {
        return;
    }

    SCALE_TYPE vec3[3] = {0};
    vec3[(int)axis] = 1.f;

    CONST_SCALE_TYPE rx = -vec3[kAxisX];
    CONST_SCALE_TYPE ry = -vec3[kAxisY];
    CONST_SCALE_TYPE rz = -vec3[kAxisZ];

    CONST_SCALE_TYPE cos_v = std::cos(radians);
    CONST_SCALE_TYPE sin_v = std::sin(radians);

    // x
    mat4[GetMat4Index(kAxisX, kAxisX)] =       1 * cos_v     + rx * rx * (1-cos_v);
    mat4[GetMat4Index(kAxisX, kAxisY)] = rx * ry * (1-cos_v) -      rz * sin_v;
    mat4[GetMat4Index(kAxisX, kAxisZ)] = rx * rz * (1-cos_v) +      ry * sin_v;

    // y
    mat4[GetMat4Index(kAxisY, kAxisX)] = ry * rx * (1-cos_v) +      rz * sin_v;
    mat4[GetMat4Index(kAxisY, kAxisY)] =       1 * cos_v     + ry * ry * (1-cos_v);
    mat4[GetMat4Index(kAxisY, kAxisZ)] = ry * rz * (1-cos_v) -      rx * sin_v;

    // z
    mat4[GetMat4Index(kAxisZ, kAxisX)] = rz * rx * (1-cos_v) -      ry * sin_v;
    mat4[GetMat4Index(kAxisZ, kAxisY)] = rz * ry * (1-cos_v) +      rx * sin_v;
    mat4[GetMat4Index(kAxisZ, kAxisZ)] =       1 * cos_v     + rz * rz * (1-cos_v);
}

// Compute the look-at matrix.
template<typename T>
inline void LookatMat4(const T* eye, const T* center, const T* up, T* mat4) {
    STATIC_ASSERT_TYPE(T);

    T* main_buf = (T*)std::malloc(sizeof(T) * 3 * 3);
    T* buf_x = main_buf + 0;
    T* buf_y = main_buf + 3;
    T* buf_z = main_buf + 6;

    SubVec3(eye, center, buf_z);   // buf_z = eye - center
    NormalizingVec3(buf_z, buf_z); // buf_z = norm(buf_z)

    CrossproductVec3(up,    buf_z, buf_x); // buf_x = cross(up,    buf_z)
    CrossproductVec3(buf_z, buf_x, buf_y); // buf_y = cross(buf_z, buf_x)

    NormalizingVec3(buf_x, buf_x); // buf_x = norm(buf_x)
    NormalizingVec3(buf_y, buf_y); // buf_y = norm(buf_y)

    // x
    mat4[GetMat4Index(kAxisX, kAxisX)] = buf_x[kAxisX];
    mat4[GetMat4Index(kAxisX, kAxisY)] = buf_y[kAxisX];
    mat4[GetMat4Index(kAxisX, kAxisZ)] = buf_z[kAxisX];

    // y
    mat4[GetMat4Index(kAxisY, kAxisX)] = buf_x[kAxisY];
    mat4[GetMat4Index(kAxisY, kAxisY)] = buf_y[kAxisY];
    mat4[GetMat4Index(kAxisY, kAxisZ)] = buf_z[kAxisY];

    // z
    mat4[GetMat4Index(kAxisZ, kAxisX)] = buf_x[kAxisZ];
    mat4[GetMat4Index(kAxisZ, kAxisY)] = buf_y[kAxisZ];
    mat4[GetMat4Index(kAxisZ, kAxisZ)] = buf_z[kAxisZ];

    // all vecs
    for (int i = 0; i < 9; ++i) {
        main_buf[i] = -main_buf[i];
    }

    // w
    mat4[GetMat4Index(kAxisW, kAxisX)] = InnerproductVec3(buf_x, eye);
    mat4[GetMat4Index(kAxisW, kAxisY)] = InnerproductVec3(buf_y, eye);
    mat4[GetMat4Index(kAxisW, kAxisZ)] = InnerproductVec3(buf_z, eye);

    mat4[GetMat4Index(kAxisX, kAxisW)] = 0.f;
    mat4[GetMat4Index(kAxisY, kAxisW)] = 0.f;
    mat4[GetMat4Index(kAxisZ, kAxisW)] = 0.f;
    mat4[GetMat4Index(kAxisW, kAxisW)] = 1.f;

    free(main_buf);
}

// Compute the perspective matrix.
template<typename T>
inline void PerspectiveMat4(CONST_SCALE_TYPE fov,
                                CONST_SCALE_TYPE aspect,
                                CONST_SCALE_TYPE near,
                                CONST_SCALE_TYPE far,
                                T *mat4) {
    CONST_SCALE_TYPE scale_y = 1.f / std::tan(ToRadians(fov) / 2.f);
    CONST_SCALE_TYPE scale_x = scale_y / aspect;
    CONST_SCALE_TYPE diff    = near - far;

    FillMat4(mat4, 0.0f); // clear

    mat4[GetMat4Index(kAxisX)] = scale_x;
    mat4[GetMat4Index(kAxisY)] = scale_y;
    mat4[GetMat4Index(kAxisZ)] = (near + far)/diff;

    mat4[GetMat4Index(kAxisZ, kAxisW)] = -1.f;
    mat4[GetMat4Index(kAxisW, kAxisZ)] = (2*far*near)/diff;
}

template<
    typename T,
    typename = std::enable_if_t<std::is_floating_point<T>::value>
>
struct Vector4 {
    Vector4() : x(0), y(0), z(0), w(0) {}
    Vector4(T scale) : x(scale), y(scale), z(scale), w(scale) {}
    Vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
    Vector4(std::initializer_list<T> list) {
        T* p = Ptr();
        for (int i = 0; i < 4; ++i) {
            *(p+i) = *(std::begin(list) + i);
        }
    }

    template<typename V>
    Vector4(const Vector4<V> &other) {
        x = other.x; y = other.y; z = other.z; w = other.w;
    }

    template<typename V>
    Vector4(const Vector4<V> &&other) {
        x = other.x; y = other.y; z = other.z; w = other.w;
    }

    inline T *GetPtr() {
        // same as Ptr()
        return (T*)(this);
    }
    inline T *Ptr() {
        return (T*)(this);
    }
    inline T& operator [](int idx) {
        return Ptr()[idx];
    }
    inline T operator [](int idx) const {
        return Ptr()[idx];
    }

    std::string ToString() {
        return Vec4ToString(Ptr());
    }

    // assign operators
    template<typename V>
    inline Vector4<T> &operator=(const Vector4<T>& rhs) {
        x = rhs.x; y = rhs.y; z = rhs.z; w = rhs.w;
        return *this;
    }

    template<typename V>
    inline Vector4<T> &operator=(const Vector4<V>&& rhs) {
        x = rhs.x; y = rhs.y; z = rhs.z; w = rhs.w;
        return *this;
    }

    T x; // 0
    T y; // 1
    T z; // 2
    T w; // 3
};

template<
    typename T,
    typename = std::enable_if_t<std::is_floating_point<T>::value>
>
struct Vector3 {
    Vector3() : x(0), y(0), z(0) {}

    Vector3(T scale) : x(scale), y(scale), z(scale) {}
    Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
    Vector3(std::initializer_list<T> list) {
        T* p = Ptr();
        for (int i = 0; i < 4; ++i) {
            *(p+i) = *(std::begin(list) + i);
        }
    }

    ~Vector3() {}

    template<typename V>
    Vector3(const Vector3<V> &other) {
        x = other.x; y = other.y; z = other.z;
    }

    template<typename V>
    Vector3(const Vector3<V> &&other) {
        x = other.x; y = other.y; z = other.z;
    }

    inline T *GetPtr() {
        // same as Ptr()
        return (T*)(this);
    }
    inline T *Ptr() {
        return (T*)(this);
    }
    inline T& operator [](int idx) {
        return Ptr()[idx];
    }
    inline T operator [](int idx) const {
        return Ptr()[idx];
    }

    std::string ToString() {
        return Vec3ToString(Ptr());
    }
    Vector4<T> ToVec4() const {
        Vector4<T> vec4(x,y,z,1);
        return vec4;
    }

    // assign operators
    template<typename V>
    inline Vector3<T> &operator=(const Vector3<V>& rhs) {
        x = rhs.x; y = rhs.y; z = rhs.z;
        return *this;
    }

    template<typename V>
    inline Vector3<T> &operator=(const Vector3<V>&& rhs) {
        x = rhs.x; y = rhs.y; z = rhs.z;
        return *this;
    }

    // plus operators
    Vector3<T> operator+(Vector3<T> &rhs) {
        Vector3<T> out;
        AddVec3(Ptr(), rhs.Ptr(), out.Ptr());
        return out;
    }
    Vector3<T> operator+(Vector3<T> &&rhs) {
        AddVec3(Ptr(), rhs.Ptr(), rhs.Ptr());
        return rhs;
    }
    template<typename V>
    friend Vector3<T> operator+(V scale, Vector3<T> &rhs) {
        Vector3<T> out;
        AddVec3(rhs.Ptr(), out.Ptr(), scale);
        return out;
    }
    template<typename V>
    friend Vector3<T> operator+(V scale, Vector3<T> &&rhs) {
        AddVec3(rhs.Ptr(), rhs.Ptr(), scale);
        return rhs;
    }
    template<typename V>
    Vector3<T> operator+(V scale) {
        Vector3<T> out;
        AddVec3(Ptr(), out.Ptr(), scale);
        return out;
    }
    Vector3<T> &operator+=(Vector3<T> &rhs) {
        AddVec3(Ptr(), rhs.Ptr(), Ptr());
        return *this;
    }
    Vector3<T> &operator+=(Vector3<T> &&rhs) {
        AddVec3(Ptr(), rhs.Ptr(), Ptr());
        return *this;
    }
    template<typename V>
    Vector3<T> &operator+=(V scale) {
        AddVec3(Ptr(), Ptr(), scale);
        return *this;
    }

    // subtraction operators
    Vector3<T> operator-(Vector3<T> &rhs) {
        Vector3<T> out;
        SubVec3(Ptr(), rhs.Ptr(), out.Ptr());
        return out;
    }
    Vector3<T> operator-(Vector3<T> &&rhs) {
        SubVec3(Ptr(), rhs.Ptr(), rhs.Ptr());
        return rhs;
    }
    template<typename V>
    friend Vector3<T> operator-(V scale, Vector3<T> &rhs) {
        Vector3<T> out;
        SubVec3(rhs.Ptr(), out.Ptr(), scale);
        return out;
    }
    template<typename V>
    friend Vector3<T> operator-(V scale, Vector3<T> &&rhs) {
        SubVec3(rhs.Ptr(), rhs.Ptr(), scale);
        return rhs;
    }
    template<typename V>
    Vector3<T> operator-(V scale) {
        Vector3<T> out;
        SubVec3(Ptr(), out.Ptr(), scale);
        return out;
    }
    Vector3<T> &operator-=(Vector3<T> &rhs) {
        SubVec3(Ptr(), rhs.Ptr(), Ptr());
        return *this;
    }
    Vector3<T> &operator-=(Vector3<T> &&rhs) {
        SubVec3(Ptr(), rhs.Ptr(), Ptr());
        return *this;
    }
    template<typename V>
    Vector3<T> &operator-=(V scale) {
        SubVec3(Ptr(), Ptr(), scale);
        return *this;
    }

    // multiplication operators
    template<typename V>
    Vector3<T> operator*(V scale) {
        Vector3<T> out;
        MulVec3(Ptr(), out.Ptr(), scale);
        return out;
    }
    template<typename V>
    friend Vector3<T> operator*(V scale, Vector3<T> &rhs) {
        Vector3<T> out;
        MulVec3(rhs.Ptr(), out.Ptr(), scale);
        return out;
    }
    template<typename V>
    friend Vector3<T> operator*(V scale, Vector3<T> &&rhs) {
        MulVec3(rhs.Ptr(), rhs.Ptr(), scale);
        return rhs;
    }
    template<typename V>
    Vector3<T> &operator*=(V scale) {
        MulVec3(Ptr(), Ptr(), scale);
        return *this;
    }

    // division operators
    template<typename V>
    Vector3<T> operator/(V scale) {
        Vector3<T> out;
        DivVec3(Ptr(), out.Ptr(), scale, false);
        return out;
    }
    template<typename V>
    Vector3<T> &operator/=(V scale) {
        MulVec3(Ptr(), Ptr(), scale, false);
        return *this;
    }

    T x; // 0
    T y; // 1
    T z; // 2
};

template<
    typename T,
    typename S = Vector4<T>,
    typename = std::enable_if_t<std::is_floating_point<T>::value>
>
struct Matrix4 {
public:
    Matrix4() {
       // all elements are zero, do nothing...
    }
    Matrix4(T scale) {
        FillDiagonalMat4(Ptr(), scale, false);
    }
    Matrix4(std::initializer_list<T> list) {
        T* p = Ptr();
        for (int i = 0; i < 16; ++i) {
            *(p+i) = *(std::begin(list) + i);
        }
    }
    Matrix4(std::initializer_list<S> list) {
        T* p = Ptr();
        for (int i = 0; i < 4; ++i) {
            sub_vec_[i] = *(std::begin(list) + i);
        }
    }

    template<typename V>
    Matrix4(const Matrix4<V> &other) {
        for (int i = 0; i < 4; ++i) sub_vec_[i] = other.sub_vec_[i];
    }

    template<typename V>
    Matrix4(const Matrix4<V> &&other) {
        for (int i = 0; i < 4; ++i) sub_vec_[i] = other.sub_vec_[i];
    }

    inline T *GetPtr() {
        // same as Ptr()
        return sub_vec_[0].GetPtr();
    }
    inline T *Ptr() {
        return sub_vec_[0].Ptr();
    }
    inline S& operator[](int idx) {
        return sub_vec_[idx];
    }
    inline S operator[](int idx) const {
        return sub_vec_[idx];
    }

    std::string ToString() { 
        return Mat4ToString(Ptr());
    }

    // assign operators
    template<typename V>
    inline Matrix4<T> &operator=(const Matrix4<V> &rhs) {
        for (int i = 0; i < 4; ++i) sub_vec_[i] = rhs.sub_vec_[i];
        return *this;
    }

    template<typename V>
    inline Matrix4<T> &operator=(const Matrix4<V> &&rhs) {
        for (int i = 0; i < 4; ++i) sub_vec_[i] = rhs.sub_vec_[i];
        return *this;
    }

    // plus operators
    Matrix4<T> operator+(Matrix4<T> &rhs) {
        Matrix4<T> out;
        AddMat4(Ptr(), rhs.Ptr(), out.Ptr());
        return out;
    }
    Matrix4<T> operator+(Matrix4<T> &&rhs) {
        AddMat4(Ptr(), rhs.Ptr(), rhs.Ptr());
        return rhs;
    }
    template<typename V>
    friend Matrix4<T> operator+(V scale, Matrix4<T> &rhs) {
        Matrix4<T> out;
        AddMat4(rhs.Ptr(), out.Ptr(), scale);
        return out;
    }
    template<typename V>
    friend Matrix4<T> operator+(V scale, Matrix4<T> &&rhs) {
        AddMat4(rhs.Ptr(), rhs.Ptr(), scale);
        return rhs;
    }
    template<typename V>
    Matrix4<T> operator+(V scale) {
        Matrix4<T> out;
        AddMat4(Ptr(), out.Ptr(), scale);
        return out;
    }

    Matrix4<T> &operator+=(Matrix4<T> &rhs) {
        AddMat4(Ptr(), rhs.Ptr(), Ptr());
        return *this;
    }
    Matrix4<T> &operator+=(Matrix4<T> &&rhs) {
        AddMat4(Ptr(), rhs.Ptr(), Ptr());
        return *this;
    }

    template<typename V>
    Matrix4<T> &operator+=(V scale) {
        AddMat4(Ptr(), Ptr(), scale);
        return *this;
    }

    // subtraction operators
    Matrix4<T> operator-(Matrix4<T> &rhs) {
        Matrix4<T> out;
        SubMat4(Ptr(), rhs.Ptr(), out.Ptr());
        return out;
    }
    Matrix4<T> operator-(Matrix4<T> &&rhs) {
        SubMat4(Ptr(), rhs.Ptr(), rhs.Ptr());
        return rhs;
    }

    template<typename V>
    Matrix4<T> operator-(V scale) {
        Matrix4<T> out;
        SubMat4(Ptr(), out.Ptr(), scale);
        return out;
    }

    Matrix4<T> &operator-=(Matrix4<T> &rhs) {
        SubMat4(Ptr(), rhs.Ptr(), Ptr());
        return *this;
    }
    Matrix4<T> &operator-=(Matrix4<T> &&rhs) {
        SubMat4(Ptr(), rhs.Ptr(), Ptr());
        return *this;
    }
    template<typename V>
    Matrix4<T> &operator-=(const V scale) {
        SubMat4(Ptr(), Ptr(), scale);
        return *this;
    }

    // multiplication operators
    Matrix4<T> operator*(Matrix4<T> &rhs) {
        Matrix4<T> out;
        MulMat4(Ptr(), rhs.Ptr(), out.Ptr());
        return out;
    }
    Matrix4<T> operator*(Matrix4<T> &&rhs) {
        MulMat4(Ptr(), rhs.Ptr(), rhs.Ptr());
        return rhs;
    }
    template<typename V>
    Matrix4<T> operator*(V scale) {
        Matrix4<T> out;
        MulMat4(Ptr(), out.Ptr(), scale);
        return out;
    }
    template<typename V>
    friend Matrix4<T> operator*(V scale, Matrix4<T> &rhs) {
        Matrix4<T> out;
        MulMat4(rhs.Ptr(), out.Ptr(), scale);
        return out;
    }
    Matrix4<T> &operator*=(Matrix4<T> &rhs) {
        MulMat4(Ptr(), rhs.Ptr(), Ptr());
        return *this;
    }
    Matrix4<T> &operator*=(Matrix4<T> &&rhs) {
        MulMat4(Ptr(), rhs.Ptr(), Ptr());
        return *this;
    }
    template<typename V>
    friend Matrix4<T> operator*(V scale, Matrix4<T> &&rhs) {
        MulMat4(rhs.Ptr(), rhs.Ptr(), scale);
        return rhs;
    }
    template<typename V>
    Matrix4<T> &operator*=(V scale) {
        MulMat4(Ptr(), Ptr(), scale);
        return *this;
    }

    Vector3<T> operator*(Vector3<T> &rhs) {
        Vector3<T> out;
        MulMat4AndVec3(Ptr(), rhs.Ptr(), out.Ptr());
        return out;
    }
    Vector3<T> operator*(Vector3<T> &&rhs) {
        Vector3<T> out;
        MulMat4AndVec3(Ptr(), rhs.Ptr(), out.Ptr());
        return out;
    }
    Vector4<T> operator*(Vector4<T> &rhs) {
        Vector4<T> out;
        MulMat4AndVec4(Ptr(), rhs.Ptr(), out.Ptr());
        return out;
    }
    Vector4<T> operator*(Vector4<T> &&rhs) {
        Vector4<T> out;
        MulMat4AndVec4(Ptr(), rhs.Ptr(), out.Ptr());
        return out;
    }

    // division operators
    template<typename V>
    Matrix4<T> operator/(V scale) {
        Matrix4<T> out;
        DivMat4(Ptr(), out.Ptr(), scale, false);
        return out;
    }
    template<typename V>
    Matrix4<T> &operator/=(V scale) {
        DivMat4(Ptr(), Ptr(), scale, false);
        return *this;
    }

    S sub_vec_[4]; // 4x4
};

using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;

using Vector4f = Vector4<float>;
using Vector4d = Vector4<double>;

using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;

const Matrix4f kIdentity_f = Matrix4f(1);
const Matrix4d kIdentity_d = Matrix4d(1);

template<typename T>
constexpr T* GetPtr(Vector3<T> &v) {
    return v.Ptr();
}
template<typename T>
constexpr T* GetPtr(Vector4<T> &v) {
    return v.Ptr();
}
template<typename T>
constexpr T* GetPtr(Matrix4<T> &m) {
    return m.Ptr();
}

// The base matrix defines the type.

template<
    typename T,
    typename V1,
    typename V2,
    typename V3,
    typename = std::enable_if_t<
                         (
                              std::is_same<std::remove_reference_t<V1>, Vector3<float>>::value &&
                              std::is_same<std::remove_reference_t<V2>, Vector3<float>>::value &&
                              std::is_same<std::remove_reference_t<V3>, Vector3<float>>::value &&
                              std::is_same<T, float>::value
                         ) || (
                              std::is_same<std::remove_reference_t<V2>, Vector3<double>>::value &&
                              std::is_same<std::remove_reference_t<V3>, Vector3<double>>::value &&
                              std::is_same<std::remove_reference_t<V1>, Vector3<double>>::value &&
                              std::is_same<T, double>::value
                         )
                       >
>
inline Matrix4<T> GetLookat(Matrix4<T> base,
                                V1&& eye,
                                V2&& center,
                                V3&& up) {
    LookatMat4(GetPtr(eye), GetPtr(center), GetPtr(up), GetPtr(base));
    return base;
}


// The base matrix defines the type.
template<typename T>
inline Matrix4<T> GetPerspective(Matrix4<T> base,
                                     CONST_SCALE_TYPE fov,
                                     CONST_SCALE_TYPE aspect,
                                     CONST_SCALE_TYPE near,
                                     CONST_SCALE_TYPE far) {
    PerspectiveMat4(fov, aspect, near, far, GetPtr(base));
    return base;
}

// The base matrix defines the type.
template<typename T>
inline Matrix4<T> GetRotation(Matrix4<T> base,
                                  const int axis,
                                  CONST_SCALE_TYPE degree) {
    RotationMat4AtAxis(GetPtr(base), axis, degree);
    return base;
}


// The in matrix defines the type.
template<
    typename T,
    typename = std::enable_if_t<
                   std::is_same<std::remove_reference_t<T>, Matrix4<float>>::value ||
                   std::is_same<std::remove_reference_t<T>, Matrix4<double>>::value
               >
>
inline std::remove_reference_t<T> Invert(T&& in) {
    std::remove_reference_t<T> out;
    InvertMat4(GetPtr(in), GetPtr(out));
    return out;
}

// The base matrix defines the type.
template<
    typename T,
    typename V,
    typename = std::enable_if_t<
                         (
                              std::is_same<std::remove_reference_t<V>, Vector3<float>>::value &&
                              std::is_same<T, float>::value
                         ) || (
                              std::is_same<std::remove_reference_t<V>, Vector3<double>>::value &&
                              std::is_same<T, double>::value
                         )
                       >
>
inline Matrix4<T> GetTranslation(Matrix4<T> base, V&& vec) {
    TranslationMat4(GetPtr(vec), GetPtr(base));
    return base;
}

#undef STATIC_ASSERT_TYPE

#undef VEC3_DIM_SIZE
#undef MAT4_DIM_SIZE

#undef CONST_SCALE_TYPE
#undef SCALE_TYPE

#endif
