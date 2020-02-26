# 滤波

## 人为增加噪声

~~~python
import cv2
import numpy as np
img = cv2.imread('C:\Users\22060\PycharmProjects\ME102\picture1')
rows,cols,chn=img.shape

for i in range(5000):
    x=np.random.randint(0,rows)
    y=np.random.randint(0,cols)
    img[x,x,:]=255

cv2.imshow("noise",img)
~~~



## Gaussian

高斯平滑也是邻域平均的思想对图像进行平滑的一种方法，在图像高斯平滑中，对图像进行平均时，不同位置的像素被赋予了不同的权重。高斯平滑与简单平滑不同，它在对邻域内像素进行平均时，给予不同位置的像素不同的权值，下图的所示的 3 * 3 和 5 * 5 领域的高斯模板。

![img](https://img-blog.csdn.net/20150606173732592)

~~~ python
result = cv2.GaussianBlur(src,(3,3),0)
titles=['Scr Image','GaussianBlur Image']
images=[src,result]
for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
~~~

## 中值滤波

在使用邻域平均法去噪的同时也使得边界变得模糊。而中值滤波是非线性的图像处理方法，在去噪的同时可以兼顾到边界信息的保留。选一个含有奇数点的窗口W，将这个窗口在图像上扫描，把窗口中所含的像素点按灰度级的升或降序排列，取位于中间的灰度值来代替该点的灰度值



## 形态学滤波

### 腐蚀与膨胀

~~~python
import cv2
import numpy as np
 
img = cv2.imread('D:/binary.bmp',0)
#OpenCV定义的结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
 
#腐蚀图像
eroded = cv2.erode(img,kernel)
#显示腐蚀后的图像
cv2.imshow("Eroded Image",eroded);
 
#膨胀图像
dilated = cv2.dilate(img,kernel)
#显示膨胀后的图像
cv2.imshow("Dilated Image",dilated);
#原图像
cv2.imshow("Origin", img)
 
#NumPy定义的结构元素
NpKernel = np.uint8(np.ones((3,3)))
Nperoded = cv2.erode(img,NpKernel)
#显示腐蚀后的图像
cv2.imshow("Eroded by NumPy kernel",Nperoded);
 
cv2.waitKey(0)
cv2.destroyAllWindows()

~~~

腐蚀既是缩小而膨胀相当于放大（在矩阵中腐蚀既是在你选择的框内判断如果有一个为0，则这个点为0；而膨胀则是有一个不为0则这个点不为0）

### 开运算和避运算

闭运算用来连接被误分为许多小块的对象，而开运算用于移除由图像噪音形成的斑点。因此，某些情况下可以连续运用这两种运算。如对一副二值图连续使用闭运算和开运算，将获得图像中的主要对象。同样，如果想消除图像中的噪声（即图像中的“小点”），也可以对图像先用开运算后用闭运算，不过这样也会消除一些破碎的对象

~~~python
import cv2
import numpy as np
 
img = cv2.imread('D:/binary.bmp',0)
#定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
 
#闭运算
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#显示腐蚀后的图像
cv2.imshow("Close",closed);
 
#开运算
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#显示腐蚀后的图像
cv2.imshow("Open", opened);
 
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~



