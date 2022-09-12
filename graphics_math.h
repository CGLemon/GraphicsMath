#ifndef GRAPHICS_MATH_H_INCLUDE
#define GRAPHICS_MATH_H_INCLUDE

#include <array>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <type_traits>

static constexpr double D_PI = 3.141592653589793238462643383279502884197f;
static constexpr float  F_PI = 3.141592653589793238462643383279502884197f;

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

#define VEC3_X_DIM (0)
#define VEC3_Y_DIM (1)
#define VEC3_Z_DIM (2)
#define SCALE_TYPE const double

// C-like functions.

template<typename T>
inline T GetX(const T* vec) {
    STATIC_ASSERT_TYPE(T);
    return vec[VEC3_X_DIM];
}
template<typename T>
inline T GetY(const T* vec) {
    STATIC_ASSERT_TYPE(T);
    return vec[VEC3_Y_DIM];
}

template<typename T>
inline T GetZ(const T* vec) {
    STATIC_ASSERT_TYPE(T);
    return vec[VEC3_Z_DIM];
}

// Convert the vector3 to std::string.
template<typename T>
inline std::string Vec3ToString(const T* vec3) {
    STATIC_ASSERT_TYPE(T);

    auto out = std::ostringstream{};
    std::cout
        << '('
        << GetX(vec3) << ", "
        << GetY(vec3) << ", "
        << GetZ(vec3) << ")";
    return out.str();
}

// Print the vector3 elements.
template<typename T>
inline void PrintVec3(const T* vec3) {
    STATIC_ASSERT_TYPE(T);

    std::cout << Vec3ToString(vec3) << std::endl;
}

// Fill the vector3.
template<typename T>
inline void FillVec3(T* vec3, SCALE_TYPE scale) {
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
inline void AddVec3(const T* in, T* out, SCALE_TYPE scale) {
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
inline void SubVec3(const T* in, T* out, SCALE_TYPE scale, bool invert) {
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
inline void MulVec3(const T* in, T* out, SCALE_TYPE scale) {
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
inline void DivVec3(const T* in, T* out, SCALE_TYPE scale, bool invert) {
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

    double val = 0.f;
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

    double scale = 1.f/std::sqrt(InnerproductVec3(in, in));
    out[VEC3_X_DIM] *= scale;
    out[VEC3_Y_DIM] *= scale;
    out[VEC3_Z_DIM] *= scale;
}

// The matrix4 multiplication operation
// wiki: https://en.wikipedia.org/wiki/Matrix_multiplication
template<typename T>
inline void MulMat4(const T* lhs, const T* rhs, T* out) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = 4;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.f;

            // TODO: loop unrolling
            for (int k = 0; k < N; k++) {
                sum += lhs[i * N + k] * rhs[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}

// Fill the diagonal elements for matrix4.
template<typename T>
inline void FillDiagonalMat4(T* mat4, T scale) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = 4;

    // TODO: loop unrolling
    for (int i = 0; i < N; i++) {
        mat4[i * N + i] = scale;
    }
}

// Fill the identity matrix4.
template<typename T>
inline void FillIdentityMat4(T* mat4) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = 4;

    // TODO: loop unrolling
    for (int i = 0; i < N*N; ++i) {
        mat4[i] = 0.f;
    }

    FillDiagonalMat4(mat4, 1.0f);
}

template<typename T>
inline void TranslationMat4(T* mat4, const T *vec3) {
    STATIC_ASSERT_TYPE(T);
    constexpr int N = 4;

    FillIdentityMat4(mat4);

    // TODO: loop unrolling
    mat4[VEC3_X_DIM * N + 3] = vec3[VEC3_X_DIM];
    mat4[VEC3_Y_DIM * N + 3] = vec3[VEC3_Y_DIM];
    mat4[VEC3_Z_DIM * N + 3] = vec3[VEC3_Z_DIM];
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

    double det = mat4[0] * inv4[0] + mat4[1] * inv4[4] + mat4[2] * inv4[8] + mat4[3] * inv4[12];
    det = 1.0f / det;

    for (int i = 0; i < 16; i++) {
        inv4[i] *= det;
    }
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

    // TODO: loop unrolling
    for (int i = 0; i < 3; ++i) {
        mat4[i * 4 + 0] = buf_x[i];
        mat4[i * 4 + 1] = buf_y[i];
        mat4[i * 4 + 2] = buf_z[i];

        buf_x[i] = -buf_x[i];
        buf_y[i] = -buf_y[i];
        buf_z[i] = -buf_z[i];
    }

    mat4[3 * 4 + 0] = InnerproductVec3(buf_x, eye);
    mat4[3 * 4 + 1] = InnerproductVec3(buf_y, eye);
    mat4[3 * 4 + 2] = InnerproductVec3(buf_z, eye);

    mat4[0 * 4 + 3] = 0;
    mat4[1 * 4 + 3] = 0;
    mat4[2 * 4 + 3] = 0;
    mat4[3 * 4 + 3] = 1.f;

    free(main_buf);
}

template<typename T, size_t N>
inline T* GetPtr(VectorBase<T, N> &vec) {
    return vec.data();
}

template<typename T, size_t N>
inline T* GetPtr(MatrixBase<T, N> &mat) {
    return mat.at(0).data();
}

template<typename T, size_t N>
inline std::string ToString(VectorBase<T, N> vec) {
    auto out = std::ostringstream{};
    out << Vec3ToString(GetPtr(vec), N);
    return out.str();
}

template<typename T, size_t N>
inline std::string ToString(MatrixBase<T, N> mat) {
    auto out = std::ostringstream{};
    out << '{';
    for (int i = 0; i < N; i++) {
        out << Vec3ToString(GetPtr(mat)+i*N, N);
        if (i != N-1) {
            out << ", ";
        }
    }
    out << '}';
    return out.str();
}

#undef STATIC_ASSERT_TYPE
#undef VEC3_X_DIM
#undef VEC3_Y_DIM
#undef VEC3_Z_DIM
#undef SCALE_TYPE

#endif
