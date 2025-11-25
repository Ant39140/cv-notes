import cv2
import os
import numpy as np

#  定义一个color库，取各个颜色的色相
COLOR_HUE = {
    "yellow": [20, 40],
    "green": [40, 95],
    "blue": [100, 125],
    "brown": [0, 20],
}

#  颜色转数字，方便输出
ColorToNum = {
    "black" : 0,
    "pink" : 1,
    "blue" : 2,
    "brown" : 3,
    "green" : 4,
    "yellow" : 5,
}

#  把一个区域内的平均hsv数值换算成颜色
def get_best_color(color):
    hsv_pixel = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv_pixel

    # print( f"H={h}, S={s}, V={v}")  ##for debug
    if v < 100 and s < 70: return "black"
    if s < 90 and v > 150 : return "pink"
    for name, (low , high) in COLOR_HUE.items():
        if low <= h <= high:
            return name
    return None

#  计算圆形区域的平均颜色
def calc_average_color(img, center , radius):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, int(radius * 0.9) , 255, -1)
    average_color = cv2.mean(img, mask=mask)
    return average_color[:3]

#  把颜色标在图片上
def label_color(img , centre , color_name):
    cv2.putText(img, color_name, centre,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

result = []
finalResult = []

#  找文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
#folder_path = os.path.join(current_dir, '..', 'data', 'ColoredBalls') ##for debug
folder_path = os.path.join(current_dir, '..', 'test')

#  错误处理
if not os.path.exists(folder_path):
        print(f"'{folder_path}' NOT FOUND")
else :
    file_list = os.listdir(folder_path)
    file_list.sort()

    for filename in file_list:
        #  读入图片
        full_path = os.path.join(folder_path, filename)
        image = cv2.imread(full_path)
        imageCopy = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #  高斯模糊一下
        blurredGray = cv2.GaussianBlur(gray, (5, 5), 0)


        #  利用HoughCircle函数找出与边缘的最佳拟合圆
        circles = cv2.HoughCircles(
            blurredGray,
            cv2.HOUGH_GRADIENT,
            dp = 1.2,
            minDist = 20,
            param1 = 150,
            param2 = 65,
            minRadius=30,
            maxRadius=50
        )


        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                #  把最佳拟合圆在图上画出来并且标出圆心
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                cv2.circle(image, (x, y), 2, (0, 0, 255), 2)
                #  找出最可能的颜色并且标出颜色
                bestColor = get_best_color(calc_average_color(imageCopy , center =(x, y), radius=r))
                label_color(image , (x-25,y+5) , bestColor)
                #  以列表形式存入结果
                colorNum = ColorToNum[bestColor]
                result.append([int(x),int(y),colorNum])
            result.sort(key = lambda x : x[2])
            finalResult += result
            result = []

            #cv2.imshow("Original", imageCopy)
            #cv2.imshow("Processed", image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

#  转np二维数组并输出
ForSummit = np.array(finalResult)
print(ForSummit.tolist())
