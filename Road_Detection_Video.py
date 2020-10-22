import cv2
import numpy as np

def Process(img):
    def region_of_mask(img, vertices):
        mask = np.zeros_like(img)
        #color_channel  = img.shape[2]
        mask_color = 255
        cv2.fillPoly(mask, vertices, mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(img, lines):
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 5)
        return img

    print(img.shape)
    height = img.shape[0]
    width = img.shape[1]
    region_of_interest_vertices = [(0,height), (width/2, height/2), (width, height)]
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(imgray, 50, 120)
    masked_img = region_of_mask(edge, np.array([region_of_interest_vertices], np.int32))
    hough = cv2.HoughLinesP(masked_img, 1, np.pi/180,100, minLineLength=20, maxLineGap=25)
    final_image = draw_lines(img, hough)
    return final_image

cap = cv2.VideoCapture('Road.mp4')
while cap.isOpened():
    ret,frame = cap.read()
    frame = Process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('d'):
        break
cap.release()
cv2.destroyAllWindows()