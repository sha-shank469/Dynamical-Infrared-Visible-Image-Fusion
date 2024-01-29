import cv2
import numpy as np
import matplotlib.pyplot as plt

class ThermalImageProcessing:

    def __init__(self,resized_ir_image,cut_visible_image,Visible_image_for_thresh_Fusion):

        self.visible_image = Visible_image_for_thresh_Fusion
        self.visible_image = cv2.cvtColor(self.visible_image, cv2.COLOR_GRAY2BGR)

        cv2.namedWindow("Original_Visible_Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Original_Visible_Image",self.visible_image)

        self.ir_image = resized_ir_image
        print("resized_ir_image_shape",self.ir_image.shape)
        cv2.namedWindow("Original_IR_Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Original_IR_Image",self.ir_image)

        self.cut_visible_img = cut_visible_image
        print("cut_visible_image_shape",self.cut_visible_img.shape)
        cv2.namedWindow("Cut_Visible_image",cv2.WINDOW_NORMAL)
        cv2.imshow("Cut_Visible_image",self.cut_visible_img)
     
# Pre-processing

    def contrast_enhancement(self):
        #stretch the image contrast
        p2,p98 = np.percentile(self.ir_image,(2,98))
        CE_image = np.uint8(cv2.normalize(self.ir_image,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U))
        Norm_CE_image = cv2.convertScaleAbs(CE_image,alpha=(255.0/(p98-p2)),beta = -255.0*p2/(p98-p2))
        cv2.namedWindow("Contrast_Enhance",cv2.WINDOW_NORMAL)
        cv2.imshow("Contrast_Enhance",Norm_CE_image)
        return Norm_CE_image 

    def sharpen_image(self,Norm_CE_image):
        #Apply unsharp masking to sharpen the image
        blurred = cv2.GaussianBlur(Norm_CE_image,(5,5),0)
        Blur_image = cv2.addWeighted(Norm_CE_image,1.5,blurred,-0.5,0)
        cv2.namedWindow("Sharpened_Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Sharpened_Image",Blur_image)
        print("Blur Image",Blur_image.shape)
        return Blur_image

    def adaptive_Histogram_Equalization(self,Blur_image):
        #apply adaptive histogram equalization to modify the image
        gray_image = cv2.cvtColor(Blur_image, cv2.COLOR_BGR2GRAY)
        # Blur_image_8u = cv2.convertScaleAbs(Blur_image)
        clahe = cv2.createCLAHE(clipLimit = 0.5,tileGridSize=(8,8))
        adaptive_histo_image = clahe.apply(gray_image)
        cv2.namedWindow("Processed_Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Processed_Image",adaptive_histo_image)
        return adaptive_histo_image

#Thresholding

    def apply_Threshold(self,threshold_value,adaptive_histo_image):
        _,threshold_image = cv2.threshold(adaptive_histo_image,threshold_value,255,cv2.THRESH_BINARY)
        print("Threshold_Image_Shape",threshold_image.shape)
        cv2.namedWindow("Threshold_Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Threshold_Image",threshold_image)
        cv2.waitKey(0)
        return threshold_image
    
    # def adaptive_threshold(self,threshold_image,block_size,constant):
    #     adaptive_threshold_image = cv2.adaptiveThreshold(threshold_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block_size,constant)
    #     cv2.namedWindow("Adaptive_Threshold_Image",cv2.WINDOW_NORMAL)
    #     cv2.imshow("Adaptive_Threshold_Image",adaptive_threshold_image)
    #     return adaptive_threshold_image

#overlay the Adaptive Threhold image over Visible Image
    def Overlay_image(self,threshold_image,alpha=0.7,beta=0.3,gamma=0):
        if threshold_image.ndim == 2:  # grayscale image
            threshold_image = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR)
        resized_threshold_image= cv2.resize(threshold_image,(self.cut_visible_img.shape[1],
                                                             self.cut_visible_img.shape[0]))
        print(resized_threshold_image.shape,self.cut_visible_img.shape)

        #APPLYING COLORMAP
        pseudocolored_image = cv2.applyColorMap(resized_threshold_image, cv2.COLORMAP_JET)
        equalized_image = cv2.equalizeHist(pseudocolored_image[:,:,2])
        pseudocolored_image[:,:,2] = equalized_image

        Overlay_image = cv2.addWeighted(self.cut_visible_img, alpha, pseudocolored_image, beta, gamma)
        print("Overlay Image",Overlay_image.shape,)

        cv2.namedWindow("Overlay_image",cv2.WINDOW_NORMAL)
        cv2.imshow("Overlay_image",Overlay_image)

        background = self.visible_image.copy()
        start = (self.visible_image.shape[0]//2 - Overlay_image.shape[0] // 2, 
                        self.visible_image.shape[1] // 2 - Overlay_image.shape[1] // 2)
        end = (self.visible_image.shape[0]//2 + Overlay_image.shape[0] // 2,
                        self.visible_image.shape[1] // 2 + Overlay_image.shape[1] // 2)
        print("start: {}, end: {}".format(start,end))
        resized_imageIR1 = cv2.resize(Overlay_image, (end[1] - start[1], end[0] - start[0]))
        background[start[0]: end[0], start[1]: end[1]] = resized_imageIR1
        cv2.imwrite("temp_thresh.jpg",background)
        cv2.namedWindow("Final_Overlayed_Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Final_Overlayed_Image",background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return Overlay_image
    
    def Overlay_Homography(self,threshold_image,overlayed_location,mask):

        if threshold_image.ndim == 2:  # grayscale image
            threshold_image = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR)
        resized_threshold_image= cv2.resize(threshold_image,(self.cut_visible_img.shape[1],
                                                             self.cut_visible_img.shape[0]))
        print(resized_threshold_image.shape,self.cut_visible_img.shape)

        #APPLYING COLORMAP
        pseudocolored_image = cv2.applyColorMap(resized_threshold_image, cv2.COLORMAP_JET)
        equalized_image = cv2.equalizeHist(pseudocolored_image[:,:,2])
        pseudocolored_image[:,:,2] = equalized_image

        Overlay_image = cv2.addWeighted(self.cut_visible_img, 0.7, pseudocolored_image, 0.3,0)
        print("Overlay Image",Overlay_image.shape)

        cv2.namedWindow("Overlay_image",cv2.WINDOW_NORMAL)
        cv2.imshow("Overlay_image",Overlay_image)

        x,y,w,h = overlayed_location
        print("OVERLAY-LOCATION",x,y,w,h)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        cv2.namedWindow("MASK",cv2.WINDOW_NORMAL)
        cv2.imshow("MASK",mask)

        bitwise_Image = cv2.bitwise_and(Overlay_image,mask)
        cv2.namedWindow("bitwise_Image",cv2.WINDOW_NORMAL)
        cv2.imshow("bitwise_Image",bitwise_Image)

        # edge_and_IR_Overlay = cv2.bitwise_and(bitwise_Image,cut_ir_image)
        # cv2.namedWindow("Edge_And_IR_Overlay",cv2.WINDOW_NORMAL)
        # cv2.imshow("Edge_And_IR_Overlay",edge_and_IR_Overlay)
        # cv2.waitKey(0)

        ir_roi = cv2.resize(bitwise_Image, (w, h))

        ir_alpha = ir_roi[:,:,0]
        cv2.namedWindow("ir_alpha",cv2.WINDOW_NORMAL)
        cv2.imshow("ir_alpha",ir_alpha)

        _, mask = cv2.threshold(ir_alpha, 0, 255, cv2.THRESH_BINARY)
        cv2.namedWindow("MASK_2",cv2.WINDOW_NORMAL)
        cv2.imshow("MASK_2",mask)

        mask_inv = cv2.bitwise_not(mask)
        cv2.namedWindow("mask_inv",cv2.WINDOW_NORMAL)
        cv2.imshow("mask_inv",mask_inv)

        background = self.visible_image.copy()

        vis_roi = background[y:y+h, x:x+w]
        cv2.namedWindow("vis_roi",cv2.WINDOW_NORMAL)
        cv2.imshow("vis_roi",vis_roi)

        ir_roi = cv2.bitwise_and(ir_roi, ir_roi, mask=mask)
        cv2.namedWindow("ir_roi_after_bitwiseand_mask",cv2.WINDOW_NORMAL)
        cv2.imshow("ir_roi_after_bitwiseand_mask",ir_roi)

        vis_roi = cv2.bitwise_and(vis_roi, vis_roi, mask=mask_inv)
        cv2.namedWindow("vis_roi_after_bitwiseand_mask",cv2.WINDOW_NORMAL)
        cv2.imshow("vis_roi_after_bitwiseand_mask",vis_roi)

        result_roi = cv2.bitwise_or(ir_roi, vis_roi)
        cv2.namedWindow("result_roi",cv2.WINDOW_NORMAL)
        cv2.imshow("result_roi",result_roi)

        background[y:y+h,x:x+w] = cv2.addWeighted(background[y:y+h,x:x+w], 0.5, result_roi, 0.5, 1)
        # background = self.visible_image.copy()
        # background[y:y+h,x:x+w] = cv2.addWeighted(background[y:y+h,x:x+w], 0.5, 
        #                                                     Overlay_image, 0.5, 0.0)
        cv2.namedWindow("Final_Overlay_Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Final_Overlay_Image",background)
        cv2.waitKey(0)
        
    def display_images(self, threshold_value):
        
        Norm_CE_image = self.contrast_enhancement()
        Blur_image = self.sharpen_image(Norm_CE_image)
        adaptive_histo_image = self.adaptive_Histogram_Equalization(Blur_image)
        threshold_image = self.apply_Threshold(threshold_value,adaptive_histo_image)
        # adaptive_threshold_image = self.adaptive_threshold(threshold_image,block_size=15,constant=2.0)
        self.Overlay_image(threshold_image)

    def display_image_homography(self,threshold_value,overlayed_location,mask):
        Norm_CE_image = self.contrast_enhancement()
        Blur_image = self.sharpen_image(Norm_CE_image)
        adaptive_histo_image = self.adaptive_Histogram_Equalization(Blur_image)
        threshold_image = self.apply_Threshold(threshold_value,adaptive_histo_image)
        
        self.Overlay_Homography(threshold_image,overlayed_location,mask)