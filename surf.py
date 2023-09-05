# opencv 3.4.2.16 (python 3.7)
import os
import cv2
import argparse
import time
from skimage import io, color, feature
import matplotlib.pyplot as plt
import numpy as np
import cv2.xfeatures2d



def surf(imageroot, result=''):
    """
    Calculate SURF features

    Args:
        imageroot (string): image file path
        result (string | optional): result path

    Return:
        pts (numpy.ndarray): Detected feature point coordinates, shape of (n, 2)

    Examples:

        from surf import surf

        imageroot = 'data/s1.jpg'
        pts = surf(imageroot, args.result)

    """

    name = imageroot.split('/')[-1] # 提取图像文件名
    img = io.imread(imageroot) # 读取图像文件，并将其存储在变量 img 中
    # gray = color.rgb2gray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像由BGR色彩空间转换为灰度色彩空间
    print('img shape', img.shape) # 打印图像的形状信息

    surf = cv2.xfeatures2d.SURF_create() # 创建SURF特征检测器对象
    kp = surf.detect(gray, None) # 使用 SURF 特征检测器在灰度图像上检测特征点，返回检测到的关键点对象
    surf_keypoints = cv2.drawKeypoints(gray, kp, outImage=None) # 在灰度图像上绘制SURF特征点，并将结果存储在变量 surf_keypoints 中
    pts = np.asarray([[p.pt[0], p.pt[1]] for p in kp]) # 提取关键点的坐标，并存储在 pts 变量中
    # print('pts', pts)
    print('pts.shape', pts.shape)

    if result:
        fig, ax = plt.subplots(1, 3, figsize=(12, 6),
                            subplot_kw=dict(xticks=[],
                                            yticks=[]))
        ax[0].imshow(gray, cmap='gray')
        ax[0].set_title('input image')
        ax[1].imshow(surf_keypoints)
        ax[1].set_title("extarcting features from image")

        cols = pts[:,0]
        rows = pts[:,1]
        ax[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[2].scatter(cols, rows)
        ax[2].set_title("keypoints coordinates")

        plt.savefig(os.path.join(result, name))
        plt.show()

    return pts
    # 将结果可视化



if __name__ == "__main__":

    #  python surf.py data/s1.jpg --result results
    imageroot = 'D:/my_pythonProject/image/1.jpg'
    pts = surf(imageroot)
    result = 'D:/my_pythonProject/image/14.jpg'
    # parser = argparse.ArgumentParser()
    # parser.add_argument("imageroot", help="image root")
    # parser.add_argument("--result", default='', type=str, help="result path")
    # args = parser.parse_args()

    if result:
        if not os.path.exists(result):
            os.mkdir(result)

    t0 = time.time()
    pts = surf(imageroot, result)
    print('surf total ', time.time() - t0)

