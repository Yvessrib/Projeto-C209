import cv2 as cv
import numpy as np

def resize(dst,img):
    width = img.shape[1]
    height = img.shape[0]
    dim = (width, height)
    resied = cv.resize(dst,dim,interpolation=cv.INTER_AREA)
    return resied

video = cv.VideoCapture(0, cv.CAP_DSHOW)
oceanVideo = cv.VideoCapture("praia.mp4")

ret, bgReference = video.read()

takeBgImage = 0

while(1):
    ret,img = video.read()
    ret2, bg = oceanVideo.read()
    
    if bg is not None:
        bg = resize(bg,bgReference)
        
    if takeBgImage == 0:
        bgReference = img
        
    diff1 = cv.subtract(img,bgReference)
    diff2 = cv.subtract(bgReference,img)
    
    diff = diff1 + diff2
    diff[abs(diff) < 25.0] = 0
    
    cv.imshow("diff1", diff)
    
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    gray[np.abs(gray) < 10] = 0
    fgMask = gray
    
    kernel = np.ones((3,3), np.uint8)
    
    fgMask = cv.erode(fgMask, kernel, iterations=2)
    fgMask = cv.dilate(fgMask, kernel, iterations=2)
    
    fgMask[fgMask > 5] = 255
    
    cv.imshow("Foreground Mask", fgMask)
    
    fgMask_inv = cv.bitwise_not(fgMask).astype(np.uint8)
    fgMask = np.uint8(fgMask)
    fgMask_inv = np.uint8(fgMask_inv)
    
    fgImage = cv.bitwise_and(img, img, mask=fgMask)
    bgImage = cv.bitwise_and(bg, bg, mask=fgMask_inv)
    
    bgSub = cv.add(bgImage,fgImage)
    
    cv.imshow("Background Removed",bgSub)
    cv.imshow("Original",img)
    
    key = cv.waitKey(5) & 0xFF
    if ord('q') == key:
        break
    elif ord('e') == key:
        takeBgImage = 1
        print("Background Captured")
    elif ord('r') == key:
        takeBgImage = 0
        print("Background Reset")
        
cv.destroyAllWindows()
video.release()
