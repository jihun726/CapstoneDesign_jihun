import cv2
cv2.namedWindow("preview")
cameraID = 0
vc = cv2.VideoCapture(cameraID, cv2.CAP_DSHOW)

if vc.isOpened(): # 첫 번째 프레임 가져오기 시도
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
