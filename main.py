#Libraries
import cv2 
import time
import mediapipe as mp 

#TODO: Capture from camera
cap = cv2.VideoCapture(0)

#TODO: Use Mediapipe FNCS
mpHand = mp.solutions.hands 
hands = mpHand.Hands(min_detection_confidence = 0.3,min_tracking_confidence = 0.7)
drawOperator = mp.solutions.drawing_utils

while True:
    #TODO: Read camera  datas frame by frame
    success, frame = cap.read()
    
    #TODO: Convert frames to RGB
    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    #TODO Exit from loop if not capture correctly
    if not success:
        break
    
    results = hands.process(imgRGB)
        
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            drawOperator.draw_landmarks(frame,handLandmarks,mpHand.HAND_CONNECTIONS)
                
            for id, landMark in enumerate(handLandmarks.landmark):
                h,w,c = frame.shape
                cx,cy = int(landMark.x*w),int(landMark.y*h)
        
        #Show the frame     
        cv2.imshow("Video Camera",frame) 
        
        #Break if q pressed
        if cv2.waitKey(1) &0xFF==ord('q'):
            break
    
#Close everything after break
cap.release()
cv2.destroyAllWindows()