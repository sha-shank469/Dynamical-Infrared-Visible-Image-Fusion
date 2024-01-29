import cv2
import sys
import numpy as np

class EdgeDetector:

    def __init__(self, Visible_image, resized_ir_image,cut_visible_image):

        self.originalImage = Visible_image
        self.image = cut_visible_image
        self.ir_image = resized_ir_image
           
        # cv2.namedWindow("cut_visible_image", cv2.WINDOW_NORMAL)
        # cv2.imshow("cut_visible_image", self.image)
        # cv2.namedWindow("Original VISIBLE image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Original VISIBLE image", self.originalImage)
        # cv2.namedWindow("Resized IR image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Resized IR image", self.ir_image)
        # cv2.waitKey(0)
    
    def grayscale(self):

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("grayscale", cv2.WINDOW_NORMAL)
        cv2.imshow("grayscale", gray)
        # cv2.imwrite("Grayscale.jpg",gray)
        return gray

    def blur(self, gray, kernal_size=(3, 3), sigma=1):
        blurred = cv2.GaussianBlur(gray, kernal_size, sigma)
        return blurred

    def gradient(self, blurred):

        gradientX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
        gradientY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
        gradientMag = cv2.magnitude(gradientX, gradientY)
        return gradientMag

    def non_max_supression(self, threshold, gradientMag):

        suppressed = cv2.threshold(gradientMag, threshold, 200, cv2.THRESH_BINARY)[1]
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        suppressed = cv2.morphologyEx(suppressed, cv2.MORPH_CLOSE, kernal)
        return suppressed

    def applycanny(self, suppressed):

        suppressed = np.uint8(suppressed)
        edges = cv2.Canny(suppressed, 25, 225)
        edges = cv2.threshold(cv2.convertScaleAbs(
            edges), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # # Apply color to the dilated edges
        color_edges = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        color_edges[np.where((color_edges != [0,0,0]).all(axis=2))] = [255,255,255]  # Set edges to green
        color_edges = cv2.bitwise_not(color_edges)
        # cv2.imwrite("CannyEdges.jpg",color_edges)
        cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
        cv2.imshow("Canny Edges", color_edges)
       
        return color_edges

    def applyLaplacian(self, image):

        mask = cv2.Laplacian(image, cv2.CV_8UC1, ksize=3)
        mask_inv = cv2.bitwise_not(mask)
        mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.threshold(mask_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
        # cv2.imwrite("LaplaceMask.jpg",mask_inv)
        cv2.namedWindow("LaplaceMask", cv2.WINDOW_NORMAL)
        cv2.imshow("LaplaceMask", mask_inv)
        cv2.waitKey(0)
        return mask_inv
    
    def User_Input(self,color_edges,mask_inv):
        while True:
            choice = input("Enter 'c' for Canny overlay and 'l' for Laplacian Overlay::")
            if choice == 'c':
                edges_to_overlay = color_edges
            elif choice == 'l':
                edges_to_overlay = mask_inv
            else:
                print("Invalid choice, TRY AGAIN")

            return edges_to_overlay
         
    def Overlay_image(self, edges_to_overlay):

        resized_edge_image = cv2.resize(edges_to_overlay,(self.ir_image.shape[1],self.ir_image.shape[0]))

        Overlay = cv2.bitwise_and(resized_edge_image, self.ir_image)
        cv2.namedWindow("Overlay-Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Overlay-Image", Overlay)
        cv2.waitKey(0)
        background = self.originalImage.copy()
        start = (self.originalImage.shape[0]//2 - Overlay.shape[0] // 2, 
                                                self.originalImage.shape[1] // 2 - Overlay.shape[1] // 2)
        end = (self.originalImage.shape[0]//2 + Overlay.shape[0] // 2, 
                                                self.originalImage.shape[1] // 2 + Overlay.shape[1] // 2)
        print("start: {}, end: {}".format(start,end))
        resized_imageIR1 = cv2.resize(Overlay, (end[1] - start[1], end[0] - start[0]))
        background[start[0]: end[0], start[1]: end[1]] = cv2.addWeighted(background[start[0]: end[0], 
                                                                                start[1]: end[1]], 0.5, resized_imageIR1, 0.5, 0.0)

        cv2.namedWindow("Overlay Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Overlay Image", background)
        cv2.waitKey(0)

        # return Overlay_image
    def Overlay_Homography(self,edges_to_overlay,overlayed_location,cut_ir_image,mask):

        # edges_to_overlay is our Processed visible image on which we have to overlay Cut_IR_Image
        # mask_inv = cv2.bitwise_not(mask)
        # mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2RGB)
        # print("MASK_INV",mask_inv.shape)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        # cv2.namedWindow("MASK",cv2.WINDOW_NORMAL)
        # cv2.imshow("MASK",mask)
        
        bitwise_Image = cv2.bitwise_and(edges_to_overlay,mask)
        cv2.namedWindow("bitwise_Image",cv2.WINDOW_NORMAL)
        cv2.imshow("bitwise_Image",bitwise_Image)
     
        edge_and_IR_Overlay = cv2.bitwise_and(bitwise_Image,cut_ir_image)
        cv2.namedWindow("Edge_And_IR_Overlay",cv2.WINDOW_NORMAL)
        cv2.imshow("Edge_And_IR_Overlay",edge_and_IR_Overlay)
   
        x,y,w,h = overlayed_location
        print("Overlay Location",x,y,w,h)

        ir_roi = cv2.resize(edge_and_IR_Overlay, (w, h))

        # Extract the alpha channel from the IR image
        ir_alpha = ir_roi[:,:,0]
        cv2.namedWindow("ir_alpha",cv2.WINDOW_NORMAL)
        cv2.imshow("ir_alpha",ir_alpha)

        # Create a mask by thresholding the alpha channel
        _, mask = cv2.threshold(ir_alpha, 0, 255, cv2.THRESH_BINARY)
        cv2.namedWindow("MASK_2",cv2.WINDOW_NORMAL)
        cv2.imshow("MASK_2",mask)

        # Invert the mask
        mask_inv = cv2.bitwise_not(mask)
        cv2.namedWindow("mask_inv",cv2.WINDOW_NORMAL)
        cv2.imshow("mask_inv",mask_inv)

        # Mask_OR = cv2.bitwise_or(mask,mask_inv)
        # cv2.namedWindow("Mask_OR",cv2.WINDOW_NORMAL)
        # cv2.imshow("Mask_OR",Mask_OR) 

        background = self.originalImage.copy()

        # Extract the visible ROI
        vis_roi = background[y:y+h, x:x+w]
        cv2.namedWindow("vis_roi",cv2.WINDOW_NORMAL)
        cv2.imshow("vis_roi",vis_roi)

        # Apply the mask to the IR ROI and the mask inverse to the visible ROI
        ir_roi = cv2.bitwise_and(ir_roi, ir_roi, mask=mask)
        cv2.namedWindow("ir_roi_after_bitwiseand_mask",cv2.WINDOW_NORMAL)
        cv2.imshow("ir_roi_after_bitwiseand_mask",ir_roi)

        vis_roi = cv2.bitwise_and(vis_roi, vis_roi, mask=mask_inv)
        cv2.namedWindow("vis_roi_after_bitwiseand_mask",cv2.WINDOW_NORMAL)
        cv2.imshow("vis_roi_after_bitwiseand_mask",vis_roi)

        # Combine the IR ROI and the visible ROI
        result_roi = cv2.add(ir_roi, vis_roi)
        cv2.namedWindow("result_roi",cv2.WINDOW_NORMAL)
        cv2.imshow("result_roi",result_roi)

        background[y:y+h,x:x+w] = cv2.addWeighted(background[y:y+h,x:x+w], 0.5, result_roi, 0.5, 1)
        cv2.namedWindow("BACKGROUND",cv2.WINDOW_NORMAL)
        cv2.imshow("BACKGROUND",background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def displayEdgeDetection(self, threshold):

        gray = self.grayscale() 

        blurred = self.blur(gray)

        gradientMag = self.gradient(blurred)

        suppressed = self.non_max_supression(threshold, gradientMag)

        edges = self.applycanny(suppressed)

        mask_inv = self.applyLaplacian(self.image)

        edges_to_overlay = self.User_Input(edges,mask_inv)

        self.Overlay_image(edges_to_overlay)

        
    def displayOverlayHomography(self, threshold,overlayed_location,cut_ir_image,mask):

        gray = self.grayscale()

        blurred = self.blur(gray)

        gradientMag = self.gradient(blurred)
       
        suppressed = self.non_max_supression(threshold, gradientMag)

        edges = self.applycanny(suppressed)

        mask_inv = self.applyLaplacian(self.image)

        edges_to_overlay = self.User_Input(edges,mask_inv)

        self.Overlay_Homography(edges_to_overlay,overlayed_location,cut_ir_image,mask)