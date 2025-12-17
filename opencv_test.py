import cv2
import time

destination  = cv2.imread('./data/destination.png', cv2.IMREAD_COLOR)
source  = cv2.imread('./data/source.png', cv2.IMREAD_COLOR)
mask = cv2.imread('./data/mask.png', cv2.IMREAD_GRAYSCALE)

topx = -35
topy = 35
x0, y0, w, h = cv2.boundingRect(mask)
px = topx + x0 + (w - 1) / 2
py = topy + y0 + (h - 1) / 2
point = (int(round(px)), int(round(py)))

st = time.time()
normal_clone = cv2.seamlessClone(source, destination, mask, point, cv2.NORMAL_CLONE)
print('OpenCV seamlessClone cv2.NORMAL_CLONE time:', time.time() - st)
cv2.imwrite('./opencv_normal_clone.png', normal_clone)

st = time.time()
mixed_clone = cv2.seamlessClone(source, destination, mask, point, cv2.MIXED_CLONE)
print('OpenCV seamlessClone cv2.MIXED_CLONE time:', time.time() - st)
cv2.imwrite('./opencv_mixed_clone.png', mixed_clone)

st = time.time()
monochrome_transfer = cv2.seamlessClone(source, destination, mask, point, cv2.MONOCHROME_TRANSFER)
print('OpenCV seamlessClone cv2.MONOCHROME_TRANSFER time:', time.time() - st)
cv2.imwrite('./opencv_monochrome_transfer.png', monochrome_transfer)