# GraphicsMath

一個 head-only 的計算機圖學數學庫

## 基本使用

Vector3 是由 x、y、z 三個數值組合而成的向量，Vector4  是 x、y、z、w 四個數值。

'''cpp
int main(int argc, char **argv) {

    Vector3f a = {1,2,3}; // 宣告 float 的 Vector3

    a += Vector3f(7,8,9); // 和其它 Vector3 相加

    std::cout << a.ToString() << std::endl; // 顯示 Vector3 的值
    std::cout << a.ToVec4().ToString() << std::endl; // 顯示 Vector4 的值

    return 0;
}
'''

<br>

Matrix4 是由四個 Vector4 組合而成的矩陣。

'''cpp
int main(int argc, char **argv) {

    Matrix4f a(8); // 宣告 float 的 Matrix4

    Matrix4f b = {
        {1,2,3,4},
        {5,6,7,8},
        {9,10,11,12},
        {13,14,15,16},
    };


    Matrix4f c = a * b; // a 和 b 做矩陣相乘
    std::cout << c.ToString() << std::endl; // 顯示 Matrix4 的值

    return 0;
}
'''
