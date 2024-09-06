import cv2
import mediapipe as mp 
import time

cap = cv2.VideoCapture("video3.mp4")

mpFaceMash = mp.solutions.face_mesh
faceMash = mpFaceMash.FaceMesh(max_num_faces=1)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius = 1)

cTime=0
pTime=0

while True:
    
    success , img = cap.read()
    if not success:
        break
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    result = faceMash.process(imgRGB)
    
    if result.multi_face_landmarks:
        
        
            for faceLms in result.multi_face_landmarks:
                mpDraw.draw_landmarks(img,faceLms,mpFaceMash.FACEMESH_TESSELATION,drawSpec,drawSpec)
                
                for id,lm in enumerate(faceLms.landmark):
                    h,w,_ = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*w)
                    print([id,cx,cy])
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "Fps :" + str(int(fps)), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
    
    cv2.imshow("img",img)
    k = cv2.waitKey(17) & 0xFF
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
cv2.destroyAllWindows()