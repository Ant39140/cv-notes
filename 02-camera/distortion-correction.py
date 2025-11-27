import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (11,8)
#  设置亚像素角点查找的停止条件（精度够高或者次数够多）
subpixel_criteria = (cv2.TERM_CRITERIA_EPS |
                     cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)

IMG_DIR = '../data/calib'


#  设置flag，设定————同时调整内参和外参，数据不好时停止计算，默认摄像机像素是正方形
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                    cv2.fisheye.CALIB_CHECK_COND + \
                    cv2.fisheye.CALIB_FIX_SKEW + \
                    cv2.fisheye.CALIB_USE_INTRINSIC_GUESS

objPoint = np.zeros((1,CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
idx = 0
for y in range(CHECKERBOARD[1]):    # 外层循环 Y (0 到 7)
    for x in range(CHECKERBOARD[0]): # 内层循环 X (0 到 10)
        objPoint[0, idx] = [x, y, 0]
        idx += 1

objPoints = []  #  储存所有图片的3D点
imgPoints = []  #  储存所有图片的2D角点

images = glob.glob(os.path.join(IMG_DIR, '*.jpg'))

if not images:
    print('No Images Found!')
    exit()

print(f"找到 {len(images)} 张图片")

valid_images = []
for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret , corners = cv2.findChessboardCorners(
        gray,  # 输入灰度图
        CHECKERBOARD,  #寻找棋盘
        cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK +
    cv2.CALIB_CB_ADAPTIVE_THRESH
        #  实现————自适应阈值（补光），归一化图像（调对比度），快速检查是否为棋盘
    )

    if ret :
        # 获取第一个角点和最后一个角点的坐标
        p0 = corners[0][0]
        p_last = corners[-1][0]
        # 计算它们距离图片左上角 (0,0) 的距离
        dist0 = p0[0] ** 2 + p0[1] ** 2
        dist_last = p_last[0] ** 2 + p_last[1] ** 2

        # 如果零点比最后一点离左上角更远，说明顺序反了（红点在右下角）
        if dist0 > dist_last:
            #print(f"图像颠倒 {image}，需翻转...")

            corners = corners[::-1].copy()
            #这里的copy可以解决Numpy只反向读取，而没有真正创建，导致后续cornerSubpix报错

        objPoints.append(objPoint)

        corners2 = cv2.cornerSubPix( gray, corners,
                    (11, 11), (-1, -1),subpixel_criteria )
        imgPoints.append(corners2)


        debug_img = img.copy()
        # 画出所有角点
        cv2.drawChessboardCorners(debug_img, CHECKERBOARD, corners2, ret)
        # 重点：画出第1个点（索引0）和最后一个点
        # 第1个点画个红圆圈
        pt_start =tuple(map(int, corners2[0].ravel()))
        cv2.circle(debug_img, pt_start,10, (0, 0, 255), -1)
        cv2.imshow("Debug Check", debug_img)
        print(f"检查 {image}：起点是否在最左上？")
        cv2.waitKey(0)

        valid_images.append(image)

    else :
        print(f" {image} Can Not Be Read")

h, w = gray.shape[:2]
if len(valid_images) > 0 :
    K = np.zeros((3, 3))
    # 手动初始化内参（全初始化为0会有神秘鱼眼相机矫正函数报错）
    # fx, fy 设为宽度的 2/3
    # cx, cy 设为图片中心
    K[0, 0] = w / 1.5
    K[1, 1] = h / 1.5
    K[0, 2] = w / 2
    K[1, 2] = h / 2
    K[2, 2] = 1.0 # 初始化内参矩阵
    D = np.zeros((4, 1))  # 初始化畸变系数

    #  定义了旋转向量和平移向量（描述相机在当时的位置）（虽然暂时用不到）
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) \
                                    for i in range(len(objPoints))]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) \
                                    for i in range(len(objPoints))]


rms , mtx , dist , rvecs , tvecs = cv2.fisheye.calibrate(
    objPoints,
    imgPoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,  #  这里直接传None也可行，因为这俩参数目前用不到
    calibration_flags ,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
)


print("Success!")
print(f"rms(均方根误差):{rms}")
print(f"K:{K}")
print(f"D:{D}")


np.savez("fisheye_calibration", K=K, D=D)
print("\n参数已保存为 'fisheye_calibration.npz'")

test_img_path = valid_images[0]
img = cv2.imread(test_img_path)
h, w = img.shape[:2]

##============注意：如果使用这个函数因为神秘原因会导致图像转换结果很搞笑===============
#  调整内参，均衡一下黑边和图像内容缺失
#  revised_K = (cv2.fisheye.estimateNewCameraMatrixForUndistortRectify
#                     (K, D, (w, h), np.eye(3), balance=0.0)
#              )


revised_K = K.copy()
# 手动缩放系数  且  系数越小 -> 视野越广
scale_factor = 0.68

revised_K[0, 0] = K[0, 0] * scale_factor   # fx
revised_K[1, 1] = K[1, 1] * scale_factor   # fy
# 保持光心在图像中心
revised_K[0, 2] = w / 2
revised_K[1, 2] = h / 2


map1, map2 = (cv2.fisheye.initUndistortRectifyMap
              (K, D, np.eye(3), revised_K, (w, h), cv2.CV_16SC2))

corrected_img = (cv2.remap
                    (img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                 )
cv2.imshow("img", img)
cv2.imshow("corrected_img", corrected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()