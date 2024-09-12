import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Coordinates: ", x, ", ", y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 0.5, (255, 255, 0), 2)
        cv2.imshow('image', img)

# Read the first frame of the video or an image file
cap = cv2.VideoCapture('rec_11_09_3.MOV')
start_time_msec = 0 * 60 * 1000 + 10 * 1000
cap.set(cv2.CAP_PROP_POS_MSEC, start_time_msec)
ret, img = cap.read()
if not ret:
    print("Can't receive frame. Exiting...")
    exit()

cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

