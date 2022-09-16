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

// basic types

template<typename T, size_t N>
using VectorBase = std::array<T, N>;

template<typename T, size_t N>
using MatrixBase = std::array<std::array<T, N>, N>;

// for Euclidean spaces types

template<typename T>
using Vector3 = VectorBase<T, 3>;
using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;

template<typename T>
using Matrix4 = MatrixBase<T, 4>;
using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;


#define STATIC_ASSERT_TYPE(T)                                  \
static_assert(std::is_same<T, float>::value ||                 \
                  std::is_same<T, double>::value,              \
                  "Only support the float and double type.");

#define VEC3_DIM_SIZE (3)
#define VEC3_X_DIM (0)
#define VEC3_Y_DIM (1)
#define VEC3_Z_DIM (2)

#define MAT4_DIM_SIZE (4)
#define MAT4_X_DIM (0) // equal to VEC3_X_DIM
#define MAT4_Y_DIM (1) // equal to VEC3_Y_DIM
#define MAT4_Z_DIM (2) // equal to VEC3_Z_DIM
#define MAT4_W_DIM (3)

#define CONST_SCALE_TYPE const double
#define SCALE_TYPE double

// C-like functions.

template<typename T>
inline T GetX(const T* vec3) {
    STATIC_ASSERT_TYPE(T);
    return vec3[VEC3_X_DIM];
}
template<typename T>
inline T GetY(const T* vec3) {
    STATIC_ASSERT_TYPE(T);
    return vec3[VEC3_Y_DIM];
}

template<typename T>
inline T GetZ(const T* vec3) {
    STATIC_ASSERT_TYPE(T);
    return vec3[VEC3_Z_DIM];
}

constexpr int GetMat4Index(const int y, const int x) {
    return y * MAT4_DIM_SIZE + x;
}

constexpr int GetMat4Index(const int dim) {
    return dim * MAT4_DIM_SIZE + dim;
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

// Convert the vector3 to std::string.
template<typename T>
inline std::string Vec3ToString(const T* vec3) {
    STATIC_ASSERT_TYPE(T);

    auto out = std::ostringstream{};
    auto FloatToString = [](T val){
        auto out = std::ostringstream{};
        int p = 6;
        int w = p + 3;
        out << std::fixed << std::setw(w) << std::setprecision(p) << val;
        return out.str();
    };

    out
        << '('
        << FloatToString(GetX(vec3)) << ", "
        << FloatToString(GetY(vec3)) << ", "
        << FloatToString(GetZ(vec3)) << ")";
    return out.str();
}

// Convert the matrix4 to std::string.
template<typename T>
inline std::string Mat4ToString(const T* mat4) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = MAT4_DIM_SIZE;

    auto out = std::ostringstream{};
    auto FloatToString = [](T val){
        auto out = std::ostringstream{};
        int p = 6;
        int w = p + 3;
        out << std::fixed << std::setw(w) << std::setprecision(p) << val;
        return out.str();
    };

    out << "{\n";
    for (int i = 0; i < N; ++i) {
        const T* offset = mat4 + i * N;
        out
            << "  ("
            << FloatToString(offset[MAT4_X_DIM]) << ", "
            << FloatToString(offset[MAT4_Y_DIM]) << ", "
            << FloatToString(offset[MAT4_Z_DIM]) << ", "
            << FloatToString(offset[MAT4_W_DIM]) << ")";
        if (i != N-1) {
            out << "\n";
        }
    }
    out << "\n}\n";
    return out.str();
}


// Print the vector3 elements.
template<typename T>
inline void PrintVec3(const T* vec3) {
    STATIC_ASSERT_TYPE(T);

    std::cout << Vec3ToString(vec3) << std::endl;
}

template<typename T>
inline void PrintMat4(const T* mat4) {
    STATIC_ASSERT_TYPE(T);

    std::cout << Mat4ToString(mat4) << std::endl;
}

// Fill the vector3.
template<typename T>
inline void FillVec3(T* vec3, CONST_SCALE_TYPE scale) {
    STATIC_ASSERT_TYPE(T);

    vec3[VEC3_X_DIM] =
        vec3[VEC3_Y_DIM] =
        vec3[VEC3_Z_DIM] =
        scale;
}

// The vector3 addition operation, [out vec] = [left vec] + [right vec].
template<typename T>
inline void AddVec3(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);

    out[VEC3_X_DIM] = lhs[VEC3_X_DIM] + rhs[VEC3_X_DIM];
    out[VEC3_Y_DIM] = lhs[VEC3_Y_DIM] + rhs[VEC3_Y_DIM];
    out[VEC3_Z_DIM] = lhs[VEC3_Z_DIM] + rhs[VEC3_Z_DIM];
}

// The vector3 and scale addition operation, [out vec] = [in vec] + scale.
template<typename T>
inline void AddVec3(const T* in, T* out, CONST_SCALE_TYPE scale) {
    STATIC_ASSERT_TYPE(T);

    out[VEC3_X_DIM] = in[VEC3_X_DIM] + scale;
    out[VEC3_Y_DIM] = in[VEC3_Y_DIM] + scale;
    out[VEC3_Z_DIM] = in[VEC3_Z_DIM] + scale;
}

// The vector3 subtraction operation, [out vec] = [left vec] - [right vec].
template<typename T>
inline void SubVec3(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);

    out[VEC3_X_DIM] = lhs[VEC3_X_DIM] - rhs[VEC3_X_DIM];
    out[VEC3_Y_DIM] = lhs[VEC3_Y_DIM] - rhs[VEC3_Y_DIM];
    out[VEC3_Z_DIM] = lhs[VEC3_Z_DIM] - rhs[VEC3_Z_DIM];
}

// The vector3 and scale subtraction operation, 
//     [out vec] = [in vec] - scale,    if not invert.
//     [out vec] =    scale - [in vec], if invert.
template<typename T>
inline void SubVec3(const T* in, T* out, CONST_SCALE_TYPE scale, bool invert) {
    STATIC_ASSERT_TYPE(T);

    if (!invert) {
        out[VEC3_X_DIM] = in[VEC3_X_DIM] - scale;
        out[VEC3_Y_DIM] = in[VEC3_Y_DIM] - scale;
        out[VEC3_Z_DIM] = in[VEC3_Z_DIM] - scale;
    } else {
        out[VEC3_X_DIM] = scale - in[VEC3_X_DIM];
        out[VEC3_Y_DIM] = scale - in[VEC3_Y_DIM];
        out[VEC3_Z_DIM] = scale - in[VEC3_Z_DIM];
    }
}

// The vector3 multiplication operation, [out vec] = [left vec] x [right vec].
template<typename T>
inline void MulVec3(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);

    out[VEC3_X_DIM] = lhs[VEC3_X_DIM] * rhs[VEC3_X_DIM];
    out[VEC3_Y_DIM] = lhs[VEC3_Y_DIM] * rhs[VEC3_Y_DIM];
    out[VEC3_Z_DIM] = lhs[VEC3_Z_DIM] * rhs[VEC3_Z_DIM];
}

// The vector3 multiplication operation, [out vec] = [in vec] x scale.
template<typename T>
inline void MulVec3(const T* in, T* out, CONST_SCALE_TYPE scale) {
    STATIC_ASSERT_TYPE(T);

    out[VEC3_X_DIM] = in[VEC3_X_DIM] * scale;
    out[VEC3_Y_DIM] = in[VEC3_Y_DIM] * scale;
    out[VEC3_Z_DIM] = in[VEC3_Z_DIM] * scale;
}

// The vector3 division operation, [out vec] = [left vec] / [right vec].
// This operation is not common.
template<typename T>
inline void DivVec3(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);

    out[VEC3_X_DIM] = lhs[VEC3_X_DIM] / rhs[VEC3_X_DIM];
    out[VEC3_Y_DIM] = lhs[VEC3_Y_DIM] / rhs[VEC3_Y_DIM];
    out[VEC3_Z_DIM] = lhs[VEC3_Z_DIM] / rhs[VEC3_Z_DIM];
}

// The vector3 and scale division operation, 
//     [out vec] = [in vec] / scale,    if not invert.
//     [out vec] =    scale / [in vec], if invert.
template<typename T>
inline void DivVec3(const T* in, T* out, CONST_SCALE_TYPE scale, bool invert) {
    STATIC_ASSERT_TYPE(T);

    if (!invert) {
        out[VEC3_X_DIM] = in[VEC3_X_DIM] / scale;
        out[VEC3_Y_DIM] = in[VEC3_Y_DIM] / scale;
        out[VEC3_Z_DIM] = in[VEC3_Z_DIM] / scale;
    } else {
        out[VEC3_X_DIM] = scale / in[VEC3_X_DIM];
        out[VEC3_Y_DIM] = scale / in[VEC3_Y_DIM];
        out[VEC3_Z_DIM] = scale / in[VEC3_Z_DIM];
    }
}

// The vector3 cross product operation.
// wiki: https://en.wikipedia.org/wiki/Cross_product
template<typename T>
inline void CrossproductVec3(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);

    out[VEC3_X_DIM] = lhs[VEC3_Y_DIM] * rhs[VEC3_Z_DIM] - lhs[VEC3_Z_DIM] * rhs[VEC3_Y_DIM];
    out[VEC3_Y_DIM] = lhs[VEC3_Z_DIM] * rhs[VEC3_X_DIM] - lhs[VEC3_X_DIM] * rhs[VEC3_Z_DIM];
    out[VEC3_Z_DIM] = lhs[VEC3_X_DIM] * rhs[VEC3_Y_DIM] - lhs[VEC3_Y_DIM] * rhs[VEC3_X_DIM];
}

// The vector3 inner product operation.
// wiki: https://en.wikipedia.org/wiki/Inner_product_space
template<typename T>
inline T InnerproductVec3(const T* lhs, const T* rhs) {
    STATIC_ASSERT_TYPE(T);

    SCALE_TYPE val = 0.f;
    val += lhs[VEC3_X_DIM] * rhs[VEC3_X_DIM];
    val += lhs[VEC3_Y_DIM] * rhs[VEC3_Y_DIM];
    val += lhs[VEC3_Z_DIM] * rhs[VEC3_Z_DIM];
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
    out[VEC3_X_DIM] *= scale;
    out[VEC3_Y_DIM] *= scale;
    out[VEC3_Z_DIM] *= scale;
}

// The matrix4 addition operation, [out mat] = [left mat] + [right mat].
template<typename T>
inline void AddMat4(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = MAT4_DIM_SIZE;

    for (int i = 0; i < N; ++i) {
        const int xx = GetMat4Index(i, MAT4_X_DIM);
        const int yy = GetMat4Index(i, MAT4_Y_DIM);
        const int zz = GetMat4Index(i, MAT4_Z_DIM);
        const int ww = GetMat4Index(i, MAT4_W_DIM);

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
        const int xx = GetMat4Index(i, MAT4_X_DIM);
        const int yy = GetMat4Index(i, MAT4_Y_DIM);
        const int zz = GetMat4Index(i, MAT4_Z_DIM);
        const int ww = GetMat4Index(i, MAT4_W_DIM);

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
        const int xx = GetMat4Index(i, MAT4_X_DIM);
        const int yy = GetMat4Index(i, MAT4_Y_DIM);
        const int zz = GetMat4Index(i, MAT4_Z_DIM);
        const int ww = GetMat4Index(i, MAT4_W_DIM);

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
        const int xx = GetMat4Index(i, MAT4_X_DIM);
        const int yy = GetMat4Index(i, MAT4_Y_DIM);
        const int zz = GetMat4Index(i, MAT4_Z_DIM);
        const int ww = GetMat4Index(i, MAT4_W_DIM);

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

            /*
            SCALE_TYPE sum = 0.f;
            for (int k = 0; k < N; k++) {
                sum += lhs[i * N + k] * rhs[k * N + j];
            }
            out[i * N + j] = sum;
            */

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
        const int xx = GetMat4Index(i, MAT4_X_DIM);
        const int yy = GetMat4Index(i, MAT4_Y_DIM);
        const int zz = GetMat4Index(i, MAT4_Z_DIM);
        const int ww = GetMat4Index(i, MAT4_W_DIM);

        out[xx] = in[xx] * scale;
        out[yy] = in[yy] * scale;
        out[zz] = in[zz] * scale;
        out[ww] = in[ww] * scale;
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
        const int xx = GetMat4Index(i, MAT4_X_DIM);
        const int yy = GetMat4Index(i, MAT4_Y_DIM);
        const int zz = GetMat4Index(i, MAT4_Z_DIM);
        const int ww = GetMat4Index(i, MAT4_W_DIM);

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
inline void FillDiagonalMat4(T* mat4, CONST_SCALE_TYPE scale) {
    STATIC_ASSERT_TYPE(T);

    FillMat4(mat4, 0.0f); // clear

    mat4[GetMat4Index(MAT4_X_DIM)] = scale;
    mat4[GetMat4Index(MAT4_Y_DIM)] = scale;
    mat4[GetMat4Index(MAT4_Z_DIM)] = scale;
    mat4[GetMat4Index(MAT4_W_DIM)] = scale;
}

// Fill the identity matrix4.
template<typename T>
inline void FillIdentityMat4(T* mat4) {
    STATIC_ASSERT_TYPE(T);

    FillDiagonalMat4(mat4, 1.0f);
}

template<typename T>
inline void TranslationMat4(T* mat4, const T *vec3) {
    STATIC_ASSERT_TYPE(T);

    FillIdentityMat4(mat4);

    mat4[GetMat4Index(MAT4_X_DIM, MAT4_W_DIM)] = vec3[VEC3_X_DIM];
    mat4[GetMat4Index(MAT4_Y_DIM, MAT4_W_DIM)] = vec3[VEC3_Y_DIM];
    mat4[GetMat4Index(MAT4_Z_DIM, MAT4_W_DIM)] = vec3[VEC3_Z_DIM];
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
inline void RotationMat4AtAxis(T* mat4, const int axis, CONST_SCALE_TYPE scale, CONST_SCALE_TYPE degree) {
    STATIC_ASSERT_TYPE(T);
    CONST_SCALE_TYPE radians = ToRadians(degree);

    FillIdentityMat4(mat4);

    if (std::abs(scale) < 1e-8f) {
        return; // return identity matrix
    }

    SCALE_TYPE vec3[3] = {0};
    vec3[axis] = 1.f;

    CONST_SCALE_TYPE rx = -vec3[VEC3_X_DIM];
    CONST_SCALE_TYPE ry = -vec3[VEC3_Y_DIM];
    CONST_SCALE_TYPE rz = -vec3[VEC3_Z_DIM];

    CONST_SCALE_TYPE cos_v = std::cos(radians);
    CONST_SCALE_TYPE sin_v = std::sin(radians);

    // x
    mat4[GetMat4Index(MAT4_X_DIM, VEC3_X_DIM)] =       1 * cos_v     + rx * rx * (1-cos_v);
    mat4[GetMat4Index(MAT4_X_DIM, VEC3_Y_DIM)] = rx * ry * (1-cos_v) -      rz * sin_v;
    mat4[GetMat4Index(MAT4_X_DIM, VEC3_Z_DIM)] = rx * rz * (1-cos_v) +      ry * sin_v;

    // y
    mat4[GetMat4Index(MAT4_Y_DIM, VEC3_X_DIM)] = ry * rx * (1-cos_v) +      rz * sin_v;
    mat4[GetMat4Index(MAT4_Y_DIM, VEC3_Y_DIM)] =       1 * cos_v     + ry * ry * (1-cos_v);
    mat4[GetMat4Index(MAT4_Y_DIM, VEC3_Z_DIM)] = ry * rz * (1-cos_v) -      rx * sin_v;

    // z
    mat4[GetMat4Index(MAT4_Z_DIM, VEC3_X_DIM)] = rz * rx * (1-cos_v) -      ry * sin_v;
    mat4[GetMat4Index(MAT4_Z_DIM, VEC3_Y_DIM)] = rz * ry * (1-cos_v) +      rx * sin_v;
    mat4[GetMat4Index(MAT4_Z_DIM, VEC3_Z_DIM)] =       1 * cos_v     + rz * rz * (1-cos_v);
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
    mat4[GetMat4Index(MAT4_X_DIM, VEC3_X_DIM)] = buf_x[VEC3_X_DIM];
    mat4[GetMat4Index(MAT4_X_DIM, VEC3_Y_DIM)] = buf_y[VEC3_X_DIM];
    mat4[GetMat4Index(MAT4_X_DIM, VEC3_Z_DIM)] = buf_z[VEC3_X_DIM];

    // y
    mat4[GetMat4Index(MAT4_Y_DIM, VEC3_X_DIM)] = buf_x[VEC3_Y_DIM];
    mat4[GetMat4Index(MAT4_Y_DIM, VEC3_Y_DIM)] = buf_y[VEC3_Y_DIM];
    mat4[GetMat4Index(MAT4_Y_DIM, VEC3_Z_DIM)] = buf_z[VEC3_Y_DIM];

    // z
    mat4[GetMat4Index(MAT4_Z_DIM, VEC3_X_DIM)] = buf_x[VEC3_Z_DIM];
    mat4[GetMat4Index(MAT4_Z_DIM, VEC3_Y_DIM)] = buf_y[VEC3_Z_DIM];
    mat4[GetMat4Index(MAT4_Z_DIM, VEC3_Z_DIM)] = buf_z[VEC3_Z_DIM];

    // all vecs
    for (int i = 0; i < 9; ++i) {
        main_buf[i] = -main_buf[i];
    }

    // w
    mat4[GetMat4Index(MAT4_W_DIM, VEC3_X_DIM)] = InnerproductVec3(buf_x, eye);
    mat4[GetMat4Index(MAT4_W_DIM, VEC3_Y_DIM)] = InnerproductVec3(buf_y, eye);
    mat4[GetMat4Index(MAT4_W_DIM, VEC3_Z_DIM)] = InnerproductVec3(buf_z, eye);

    mat4[GetMat4Index(MAT4_X_DIM, MAT4_W_DIM)] = 0.f;
    mat4[GetMat4Index(MAT4_Y_DIM, MAT4_W_DIM)] = 0.f;
    mat4[GetMat4Index(MAT4_Z_DIM, MAT4_W_DIM)] = 0.f;
    mat4[GetMat4Index(MAT4_W_DIM, MAT4_W_DIM)] = 1.f;

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

    mat4[GetMat4Index(MAT4_X_DIM)] = scale_x;
    mat4[GetMat4Index(MAT4_Y_DIM)] = scale_y;
    mat4[GetMat4Index(MAT4_Z_DIM)] = (near + far)/diff;
    mat4[GetMat4Index(MAT4_Z_DIM, MAT4_W_DIM)] = -1.f;
    mat4[GetMat4Index(MAT4_W_DIM, MAT4_Z_DIM)] = (2*far*near)/diff;
}

template<typename T, size_t N>
inline T* GetPtr(VectorBase<T, N> &vec) {
    return vec.data();
}

template<typename T, size_t N>
inline T* GetPtr(MatrixBase<T, N> &mat) {
    return mat.at(0).data();
}

template<typename T>
inline std::string ToString(Vector3<T> vec3) {
    return Vec3ToString(GetPtr(vec3));
}

template<typename T>
inline std::string ToString(Matrix4<T> mat4) { 
    return Mat4ToString(GetPtr(mat4));
}

#undef STATIC_ASSERT_TYPE

#undef VEC3_DIM_SIZE
#undef VEC3_X_DIM
#undef VEC3_Y_DIM
#undef VEC3_Z_DIM

#undef MAT4_DIM_SIZE
#undef MAT4_X_DIM
#undef MAT4_Y_DIM
#undef MAT4_Z_DIM
#undef MAT4_W_DIM

#undef CONST_SCALE_TYPE
#undef SCALE_TYPE

#endif
