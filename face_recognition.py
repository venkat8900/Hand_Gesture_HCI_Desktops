# import required libraries

import tensorflow.compat.v1 as tf
import numpy as np
import os,cv2
import sys,argparse
from glob import glob
import time
from pynput.mouse import Button ,Controller
import wx

rec = False

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

countr = 0
# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check the ID if exist 
        if(Id == 1):
            Id = "{0:.2f}%".format(round(100 - confidence, 2))
        if 100 - confidence >= 10:
            countr += 1

        # Put text describe who is in the picture
        #cv2.rectangle(im, (x-22,y-90), (x+22, y-22), (0,255,0), -1)
        #cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im)
    if countr >= 50:
        rec = True
        break

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()

cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

isBgCaptured = 0

c_frame=-1
p_frame=-1

#Setting threshold for number of frames to compare
thresholdframes=50

tf.disable_v2_behavior()

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('E:\Hand-Gesture-Recognition-Using-CNN-master\Hand-Gesture-Recognition-Using-CNN-master/handgest_1.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 10)) 


#Real Time prediction
def predict(frame,y_test_images):
    image_size=50
    num_channels=3 # RGB channels
    images = []
    image=frame
    cv2.imshow('test',image)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)

    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)

    ### Creating the feed_dict that is required to be fed to calculate y_pred 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of gest0,......,probability_of_gest9]
    return np.array(result)


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

if rec == True:  
#Open Camera object
    cap = cv2.VideoCapture(0)

#Decrease frame size (4=width,5=height)
    cap.set(4, 700)
    cap.set(5, 400)

    h,s,v = 150,150,150
    i=0
    counter = 0
    while(i<1000000):
        ret, frame = cap.read()
            
        cv2.rectangle(frame, (300,300), (100,100), (0,255,0),0)
        crop_frame=frame[100:300,100:300]
     #Blur the image
    #blur = cv2.blur(crop_frame,(3,3))
        blur = cv2.GaussianBlur(crop_frame, (3,3), 0)
        
    #Convert to HSV color space
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    
        cv2.imshow('main',frame)
    
        """Modded"""
        if isBgCaptured == 1:  # this part wont run until background captured
            imge = removeBG(frame)
            
            imge = imge[100:300,100:300]  # clip the ROI
            cv2.imshow('mask', imge)

    # convert the image into binary image
            gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    #cv2.imshow('blur', blur)
            ret, med = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('ori', med)
    
    ##Displaying frames
        
    #cv2.imshow('masked',med)

    ##resizing the image
            med=cv2.resize(med,(50,50))
    ##Making it 3 channel
            med=np.stack((med,)*3)
    ##adjusting rows,columns as per x
            med=np.rollaxis(med,axis=1,start=0)
            med=np.rollaxis(med,axis=2,start=0)
    ##Rotating and flipping correctly as per training image
            M = cv2.getRotationMatrix2D((25,25),270,1)
            med = cv2.warpAffine(med,M,(50,50))
            med=np.fliplr(med)
            ##converting expo to float
            np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    ##printing index of max prob value
            ans=predict(med,y_test_images)
    #print(ans)
    #print(np.argmax(max(ans)))
    #Comparing for 50 continuous frames
            c_frame=np.argmax(max(ans))
            if(c_frame==p_frame):
                counter=counter+1
                p_frame=c_frame
                if (counter==thresholdframes):
                    print(ans)
                    print("Gesture:"+str(c_frame))
                    if c_frame == 1:
                        break
                    elif c_frame == 2:
                
                        mouse=Controller()

                        app=wx.App(False)
                        (sx,sy)=wx.GetDisplaySize()
                        (camx,camy)=(320,240)
                        #cap1=cv2.VideoCapture(0)
                        #cap1.set(3,camx)
                        #cap1.set(4,camy)

                        #range for HSV (green color)
                        lower_g=np.array([33,70,30])
                        upper_g=np.array([102,255,255])

                        #red color
                        lower_r1 = np.array([0,50,50])
                        upper_r1 = np.array([10,255,255])
                        
                        lower_r2 = np.array([170,50,50])
                        upper_r2 = np.array([180,255,255])

                        #Kerenel
                        kernelOpen=np.ones((5,5))
                        kernelClose=np.ones((20,20))

                        mLocOld=np.array([0,0])
                        mouseLoc=np.array([0,0])

                        DampingFactor=2 #Damping factor must be greater than 1

                        isPressed=0
                        openx,openy,openw,openh=(0,0,0,0)
                        
                        countrr = 0

                        while True:

                            ret,img=cap.read()

                            img=cv2.resize(img,(340,220))
                            imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                            mask=cv2.inRange(imgHSV,lower_g,upper_g)

                            #for red
                            mask0=cv2.inRange(imgHSV,lower_r1,upper_r1)
                            mask1=cv2.inRange(imgHSV,lower_r2,upper_r2)
                        
                            maskr = mask0+mask1

                            #using morphology to erase noise as maximum as possible 
                            new_mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
                            another_mask=cv2.morphologyEx(new_mask,cv2.MORPH_CLOSE,kernelClose)
                            final_mask=another_mask

                            #red
                            new_mask=cv2.morphologyEx(maskr,cv2.MORPH_OPEN,kernelOpen)
                            another_mask=cv2.morphologyEx(new_mask,cv2.MORPH_CLOSE,kernelClose)
                            final_maskr=another_mask
    
                            #im2,conts,h=cv2.findContours(final_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                            _,conts,h=cv2.findContours(final_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                            #red
                            _,contsr,hr=cv2.findContours(final_maskr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                            # Once 2 objects are detected the center of there distance will be the reference on controlling the mouse
                            if(len(conts)==2):

                                #if the button is pressed we need to release it first
                                if(isPressed==1):
                                    isPressed=0
                                    mouse.release(Button.left)

                                #drawing the rectagle around both objects
                                x1,y1,w1,h1=cv2.boundingRect(conts[0])
                                x2,y2,w2,h2=cv2.boundingRect(conts[1])
                                cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
                                cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)

                                #the line between the center of the previous rectangles
                                cx1=int(x1+w1/2)
                                cy1=int(y1+h1/2)
                                cx2=int(x2+w2/2)
                                cy2=int(y2+h2/2)
                                cv2.line(img,(cx1,cy1),(cx2,cy2),(255,0,0),2)

                                #the center of that line (reference point)
                                clx=int((cx1+cx2)/2)
                                cly=int((cy1+cy2)/2)
                                cv2.circle(img,(clx,cly),2,(0,0,255),2)

                                #adding the damping factor so that the movement of the mouse is smoother
                                mouseLoc=mLocOld+((clx,cly)-mLocOld)/DampingFactor
                                mouse.position=(sx-int((mouseLoc[0]*sx)/camx),int((mouseLoc[1]*sy)/camy))
                                while mouse.position!=(sx-int((mouseLoc[0]*sx)/camx),int((mouseLoc[1]*sy)/camy)):
                                    pass

                                #setting the old location to the current mouse location
                                mLocOld=mouseLoc

                                #these variables were added so that we get the outer rectangle that combines both objects 
                                openx,openy,openw,openh=cv2.boundingRect(np.array([[[x1,y1],[x1+w1,y1+h1],[x2,y2],[x2+w2,y2+h2]]]))

                            #when there's only when object detected it will act as a left click mouse    
                            elif(len(conts)==1):
                                x,y,w,h=cv2.boundingRect(conts[0])

                                # we check first and we allow the press fct if it's not pressed yet
                                #we did that to avoid the continues pressing 
                                if(isPressed==0):
                                    
                                    if(abs((w*h-openw*openh)*100/(w*h))<30): #the difference between th combined rectangle for both objct and the 
                                        isPressed=1                          #the outer rectangle is not more than 30%
                                        mouse.press(Button.left)
                                        openx,openy,openw,openh=(0,0,0,0)

                                #this else was added so that if there's only one object detected it will not wwwwwwwwwwwwwwwwwwwgetting rectangle coordinates and drawing it 
                                x,y,w,h=cv2.boundingRect(conts[0])
                                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

                                #getting the center of the circle that will be inside the outer rectangle
                                cx=int(x+w/2)
                                cy=int(y+h/2)
                                cv2.circle(img,(cx,cy),int((w+h)/4),(0,0,255),2)#drawing that circle
                                
                                mouseLoc=mLocOld+((cx,cy)-mLocOld)/DampingFactor
                                mouse.position=(sx-int((mouseLoc[0]*sx)/camx),int((mouseLoc[1]*sy)/camy))
                                while mouse.position!=(sx-int((mouseLoc[0]*sx)/camx),int((mouseLoc[1]*sy)/camy)):
                                    pass
                                mLocOld=mouseLoc
                            elif(len(conts)==3):
                                countrr += 1
                                if countrr == 30:
                                    break
                                
                            #showing the results 
                            cv2.imshow("Virtual mouse",img)
                            
                            #waiting for 'W' to be pressed to quit 
                            if cv2.waitKey(1) & 0xFF==ord('w'):
                                break

                        cv2.destroyWindow('Virtual mouse')
                        del app
                        break
                    elif c_frame == 4:
                        os.startfile("notepad.exe")
                    elif c_frame == 5:
                        os.startfile("ubuntu.exe") # give additonal commands based on requirements
                    counter=0
                    i=0
            else:
                p_frame=c_frame
                counter=0
    
 #close the output video by pressing 'ESC'
        k = cv2.waitKey(2) & 0xFF
        if k == 27:
            break
        elif k == ord('b'):  # press 'b' to capture the background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = 1
            print( '!!!Background Captured!!!')
        i=i+1
    
    cap.release()
    cv2.destroyAllWindows()
