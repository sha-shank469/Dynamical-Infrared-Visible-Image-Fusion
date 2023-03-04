import math
import cv2
import numpy as np
#import decimal
def calculate_distance(alphaVisible,alphaIR,dm,beta,tx=0,tz=0.01):

    #alphavisible is FOV of visible camera radians 
    #alphaIR is FOV of IR camera in radians 
    #beta is pitch of IR camera 
    #dm is distance measured by telemeter in meter
    #tx is half of the physical size in meters
    #tz is the horizontal distance between both the camera IR and Visible
    #tx is the vertical distance at which the visible camera is placed 

    #if isinstance(imageVisible,str):

    imageVisible = cv2.imread("/home/x2gointern/Documents/DJIPicsForFusion/DJI_0023_W.JPG")

    #if isinstance(imageIR,str):

    imageIR = cv2.imread("/home/x2gointern/Documents/DJIPicsForFusion/DJI_0024_T.JPG")

    # cv2.imshow("Visible image",imageVisible)
    # cv2.imshow("IR Image",imageIR)
    # cv2.waitKey(0)
    

    d = dm*math.cos(beta)
    print("d:",d)
    dVisible = 2*(d-tx)*math.tan(alphaVisible/2)
    dIR = d*(math.tan(alphaIR/2 + beta) + math.tan(alphaIR/2-beta))
    
    # print(alphaVisible)
    # print(alphaIR)

    #resizing the visible image equal to Infrared image 
    heightIR,widthIR,_ = imageIR.shape
    print("Original Height IR:",heightIR)
    print("Original Width IR:",widthIR)
    heightVisible,widthVisible,_ = imageVisible.shape
    print("Original Height Visible:",heightVisible)
    print("Original width Visible:",widthVisible)



    if alphaVisible >= alphaIR:

        #Resizing the visible
        heightVis = int(heightIR*(alphaVisible/alphaIR))
        print("Resized Visible Image Height:",heightVis)
        widthVis = int(widthIR*(alphaVisible/alphaIR))
        print("Resized Visible Image Width",widthVis)
        imageVisible1 = cv2.resize(imageVisible,(widthVis,heightVis))
        cv2.imshow("imageVisible1",imageVisible1)
        cv2.waitKey(0)

        #Resizing the IR image
        new_width = int(widthIR*(dIR/dVisible)*(heightVis/heightIR))
        print("Resized IR Image Width:",new_width)
        new_height = int(heightVis * (dIR/dVisible))
        print("Resized IR image Height:",new_height)
        imageIR1 = cv2.resize(imageIR,(new_width,new_height))
        cv2.imshow("Resized IR Image",imageIR1)
        cv2.waitKey(0)


        
        #padding the ir image
        #heightIR,widthIR,_ = imageIR1.shape
        desired_height, desired_width = heightVis, widthVis

        # Get the current height and width of the IR image
        current_height, current_width, _ = imageIR1.shape

        # Calculate the amount of padding required on each side
        pad_top = int((desired_height - current_height) / 2)
        pad_bottom = desired_height - current_height - pad_top
        pad_left = int((desired_width - current_width) / 2)
        pad_right = desired_width - current_width - pad_left

        # Pad the IR image with zeros to match the desired size
        imageIR2 = np.pad(imageIR1, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')

        cv2.imshow("Padded IR imageIR",imageIR2)
        cv2.waitKey(0)

        # (int(widthIR*scaling_factor*scaling_factor1),int(heightVisible*scaling_factor2))
        # imageVisible2 = cv2.resize(imageVisible1,(imageIR.shape[1],imageIR.shape[0]))
        # widthVisible1,heightVisible1,_ = imageVisible2.shape
        # print("Width of visible after Resize",widthVisible1)
        # print("Height of  Visible after Resize",imageVisible1)
        # imageIR = cv2.resize(imageIR,(imageVisible.shape[1],imageVisible.shape[0]))



        tz_prime = d*math.tan(beta)
        print("tz_prime",tz_prime)
        y = ((heightVis/2)*(tz-tz_prime))/((d-tx)*math.tan(alphaVisible/2))
        print("y:",y)


        if y<1:
        #calculate the vertical position of the top and bottom edges of the IR image
            top_y = int(y - (tz_prime / (d - tx)) * (heightVis/2))
            bottom_y = top_y + imageIR2.shape[0]

            # ensure the IR image is within bounds of the visible image
            if top_y < 0:
                offset = -top_y
                top_y = 0
            else:
                offset = 0

            if bottom_y > imageVisible1.shape[0]:
                imageIR = imageIR2[:imageVisible1.shape[0]-top_y, :]
            else:

                imageIR = imageIR2[offset:,:]

            # create background image and place IR and visible images on it
            background = np.zeros_like(imageVisible1)
            background[top_y:bottom_y, :] = imageIR
            background[(background.shape[0]-imageVisible1.shape[0])//2:(background.shape[0]+imageVisible1.shape[0])//2,:] += imageVisible1
            imageIR_resized = cv2.resize(imageIR2, (imageVisible1.shape[1], imageVisible1.shape[0]))
            background = cv2.addWeighted(imageIR_resized, 0.5, imageVisible1, 0.5, 0.0)
            cv2.imshow("Overlap image",background)
            cv2.waitKey(0)   
        elif y>=1:
            result = str(y).split('.')[1]
            y = '.' + result
            y = float(y)
            top_y = float(y - (tz_prime / (d - tx)) * (heightVis/2))
            print(top_y)
            bottom_y = top_y + imageIR2.shape[0]
            print(bottom_y)


            top_y_int = int(top_y)
            bottom_y_int = int(bottom_y)

            # ensure the IR image is within bounds of the visible image
            if top_y < 0:
                offset = -top_y
                top_y = 0
            else:
                offset = 0

            
            if bottom_y > imageVisible1.shape[0]:
                imageIR = imageIR2[:imageVisible1.shape[0]-top_y_int, :]
            else:

                imageIR = imageIR2[offset:,:]

            #create background image and place IR and visible images on it
            background = np.zeros_like(imageVisible1)
            background[top_y_int:bottom_y_int, :] = imageIR
            background[(background.shape[0]-imageVisible1.shape[0])//2:(background.shape[0]+imageVisible1.shape[0])//2,:] += imageVisible1
            imageIR_resized = cv2.resize(imageIR2, (imageVisible1.shape[1], imageVisible1.shape[0]))
            background = cv2.addWeighted(imageIR_resized, 0.5, imageVisible1, 0.5, 0.0)
            cv2.imshow("Overlap image",background)
            cv2.waitKey(0)

    else: 
        print('Error')
        
    ##scale the distance based on image size
    # dIR *= widthIR
    # dVisible *= widthIR 
    return(dIR,dVisible) # return a tuple containing the distance observed by IR camera and visible camera

    # def calculate_ir_localization(heightVisible,tz,d,tx,alphavisible,beta):
    #     tzprime = d*math.tan(beta)
    #     y = ((heightVisible/2)*(tz-tzprime))/(d-tx)*math.tan(alphavisible/2)
        

def main():

    dir,dvisible = calculate_distance(math.radians(67.2),math.radians(48),6.8,math.radians(0))
    print("Distance observed by IR camera:",dir ,"meters")
    print("Distance observed by visible camera:",dvisible,"meters")
    
main()




