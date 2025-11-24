import cv2
import numpy as np

image = cv2.imread("../data/ColoredBalls/1_2.png")
imageCopy = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


gray = cv2.GaussianBlur(gray, (5, 5), 0)

circles = cv2.HoughCircles(
    gray,
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
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 2, (0, 0, 255), 2)

cv2.imshow("Original", imageCopy)
cv2.imshow("Processed", image)
cv2.waitKey(0)
cv2.destroyAllWindows()