# 霍夫变换

## 基本介绍

霍夫变换的基本思想是点 —— 线的对偶性(duality)，当然这只不过是所有人都这样说了，所谓线的对偶性就是线与线的某种变换之后的对应或者映射关系，实际上霍夫变换是直角坐标系空间线与变换域空间的线的映射关系，考虑傅里叶变换。

即利用公式：
$$
y=px+q
$$
得到通过某一点的所以直线，然后判断其他点是否与该线共线，找出其峰值的那条直线即是我们要的直线。

具体实现步骤：

1)、构造一个P、Q空间的二维累加数组A(p,q)

2)、从f(x,y)的指定区域中取(xi,yi)，按方程q=-pxi+yi在[pmin,pmax]中遍取可能的p值计算得到可能的q值。

3)、在对应的位置计算A(p,q) =A(p,q)+1

4)、重复2)、3)直到将从f(x,y)的指定区域中的所有点取完。此时，A(p,q)数组中最大值所对应的p，q就是方程y=px+q中的p、q值。

5)、根据y=px+q绘出f(x,y)中的直线



## 边缘检测

~~~ python
import cv2
import numpy as np

img = cv2.imread("picture1.JPG",cv2.IMREAD_UNCHANGED)

img = cv2.GaussianBlur(img, (3, 3), 0)
canny = cv2.Canny(img, 50, 150)

cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

<img src="HOUGH.assets/image-20200225005612333.png" alt="image-20200225005612333" style="zoom: 10%;" />

可调程序

~~~ python
import cv2
import numpy as np


def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
    dst = cv2.bitwise_and(img01, img01, mask=detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo', dst)


lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img01 = cv2.imread("picture1.JPG",cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img01, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny demo')

cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)

CannyThreshold(0)  # initialization
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
~~~

<img src="HOUGH.assets/image-20200225005755431.png" alt="image-20200225005755431" style="zoom:10%;" />

##  直线检测

~~~ python
import cv2
import numpy as np

img = cv2.imread("picture1.JPG",cv2.IMREAD_UNCHANGED)

img = cv2.GaussianBlur(img, (3, 3), 0)
canny = cv2.Canny(img, 50, 150)

cv2.imshow('Canny', canny)
# 标准霍夫线变换
def line_detection_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)  # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的
        a = np.cos(theta)   # theta是弧度
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))  # 直线起点横坐标
        y1 = int(y0 + 1000 * (a))   # 直线起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 直线终点横坐标
        y2 = int(y0 - 1000 * (a))   # 直线终点纵坐标
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("image_lines", image)


# 统计概率霍夫线变换
def line_detect_possible_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("line_detect_possible_demo", image)


if __name__ == "__main__":
    img = cv2.imread("text1.png")
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("input image", img)
    line_detect_possible_demo(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

<img src="HOUGH.assets/image-20200225010840525.png" alt="image-20200225010840525" style="zoom:10%;" />