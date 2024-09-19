# Graphics Math

一個 head-only 的計算機圖學數學庫，實做了一些常用功能，無須任何第三方的庫。

## 需求

支援 C++14 或以上的編譯器。

## 基本使用

```cpp
int main(int argc, char **argv) {

    Vector3f a = {1,2,3}; // 宣告 float 的 Vector3

    a += Vector3f(7,8,9); // 和其它 Vector3 相加

    std::cout << a << std::endl; // 顯示 Vector3 的值

    auto b = a.Cast<Vector4d>(); // 傳換成 Vector4 ，並且是 double 型態

    return 0;
}
```

<br>


```cpp
int main(int argc, char **argv) {

    // 宣告 float 的 Matrix4，對角線數值為 8
    Matrix4f a(8);

    Matrix4f b = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10,11,12,
        13,14,15,16,
    };

    Matrix4f c = a * b; // a 和 b 做矩陣相乘
    std::cout << c << std::endl; // 顯示 Matrix4 的值
    return 0;
}
```

## 常用的圖學矩陣

#### 單位矩陣

```cpp
int main(int argc, char **argv) {

    Matrix4f mat4(1);
}
```

#### 反矩陣

```cpp
int main(int argc, char **argv) {

    Matrix4f mat4(1);

    auto invert = mat4.Invert();
}
```

#### 轉移矩陣

```cpp
int main(int argc, char **argv) {

    auto rot = Matrix4f::GetTranslation(
                   Vector3f(1,2,3) // 轉移向量
               );
}
```


#### 旋轉矩陣

```cpp
int main(int argc, char **argv) {

    auto rot = Matrix4f::GetRotation(
                   kAxisX, // 轉軸
                   60.f    // 旋轉角度（degree）
               );
}
```

#### Look At 矩陣

```cpp
int main(int argc, char **argv) {

    auto lookat = Matrix4f::GetLookat(
                      Vector3f(-1, 0, -1.4), // 攝影機位置
                      Vector3f( 0, 0, 0),    // 拍攝的位置
                      Vector3f( 5, 1, 0)     // 頭頂方向
                  );
}
```

#### 透視矩陣

```cpp
int main(int argc, char **argv) {

    auto Perspective = Matrix4f::GetPerspective(
                           45.0f,      // 視角（degree）
                           1,          // 寬高比
                           0.1f,       // 近平面
                           100.0f      // 遠平面
                       );
}
```

