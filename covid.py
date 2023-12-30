import numpy as np
import cv2 as cv
import cv2

img = cv.imread('covi19_SARS_microscope.png')
cv.imshow('Imagem Original', img)

alpha = 1.0 #contrast control
beta = 30 #brightness control

ajuste =  cv.convertScaleAbs(img, alpha = alpha, beta = beta)
#cv.imshow('controle brilho e contraste', ajuste)

gray = cv.cvtColor(ajuste, cv.COLOR_BGR2GRAY)
#cv.imshow('imagem cinza', gray)

blur1 = cv.GaussianBlur(gray, (3, 3), 0)
#cv.imshow('embaçar 3x3', blur1)

threshold, thresh = cv.threshold(blur1, 150, 200, cv.THRESH_BINARY)
#cv.imshow('threshold normal', thresh)

canny = cv.Canny(thresh, 150, 100)
#cv.imshow('filtro canny',canny)

ret, thresh = cv.threshold(canny, 100, 255, cv.ADAPTIVE_THRESH_MEAN_C + cv.THRESH_OTSU)
kernel = np.ones((2, 2), np.uint8)
closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations = 1)
dilating = cv.dilate(closing, kernel, iterations = 2)
dst = cv.Canny(dilating, 100, 255, None, 3)
imagem_binaria_sobel = np.where(dilating > dst, 255, 0).astype(np.uint8)
cv.imshow('Imagem Modificada', imagem_binaria_sobel)

cv.waitKey(0)
cv2.destroyAllWindows()