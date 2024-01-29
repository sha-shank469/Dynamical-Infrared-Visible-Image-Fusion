import cv2
import numpy as np

class ImageRegistration:
    def __init__(self, homography_matrix):
        self.homography_matrix = homography_matrix

    def register_image(self, visible_image, ir_image):
        Visible_image = visible_image.copy()
        cv2.imwrite("Visible_image.jpg",Visible_image)
        cv2.namedWindow("Visible_image---", cv2.WINDOW_NORMAL)
        cv2.imshow("Visible_image---", Visible_image)

        ir_image = ir_image.copy()
        cv2.namedWindow("IR_image---", cv2.WINDOW_NORMAL)
        cv2.imshow("IR_image---", ir_image)

        heightVisible, widthVisible,_ = visible_image.shape
        heightIR, widthIR,_ = ir_image.shape
 
        if heightVisible < heightIR or widthVisible < widthIR:
            raise ValueError(
                "Large Image, Dimension should be greater than or Equal to small Dimension Image")

        # wrapImage = Visible_image_gray.copy()
        warpImage = cv2.warpPerspective(ir_image, M=self.homography_matrix,
                                        dsize=(widthVisible, heightVisible))
    
        print("WarpImage",warpImage.shape)
        cv2.imwrite("WarpedImage.jpg",warpImage)
        cv2.namedWindow("warpped Image", cv2.WINDOW_NORMAL)
        cv2.imshow("warpped Image", warpImage)

        points_IR = np.array([[[0,0],[0,ir_image.shape[0]],
                               [ir_image.shape[1],ir_image.shape[0]],
                               [ir_image.shape[1],0]]],dtype=np.float32)
        print("points_IR",points_IR)

        points_warped = cv2.perspectiveTransform(points_IR,m=self.homography_matrix)
        print("points_warped : ",points_warped)

        min_x = int(np.min(points_warped[:,:,0]))
        max_x = int(np.max(points_warped[:,:,0]))
        min_y = int(np.min(points_warped[:,:,1]))
        max_y = int(np.max(points_warped[:,:,1]))
        print(min_x,max_x,min_y,max_y)

        # min_x = max(min_x, 0)
        # max_x = min(max_x, warpImage.shape[1])
        # min_y = max(min_y, 0)
        # max_y = min(max_y, warpImage.shape[0])
        # print(min_x,max_x,min_y,max_y)

# Cutting IR image from the Warped image and creating a mask.

        # cut_ir_image = warpImage[min_y:max_y, min_x:max_x]
        # cv2.imwrite("cut_ir_image.png",cut_ir_image)
        # print("cut_ir_image :",cut_ir_image.shape)
        if min_y >= 0 and max_y <= warpImage.shape[0] and min_x >= 0 and max_x <= warpImage.shape[1]:
            # mask = np.zeros_like(cut_ir_image[:,:,0])
            cut_ir_image = warpImage[min_y:max_y, min_x:max_x]
            mask = np.ones_like(cut_ir_image[:,:,0]) * 255
        else:
            # points = np.int32(points_warped).reshape(-1,1,2)
            # print("POINTS",points)
            # mask = cv2.fillPoly(mask, [points], (255,255,255))
            min_x = max(min_x, 0)
            max_x = min(max_x, warpImage.shape[1])
            min_y = max(min_y, 0)
            max_y = min(max_y, warpImage.shape[0])
            cut_ir_image = warpImage[min_y:max_y, min_x:max_x]
            print(min_x,max_x,min_y,max_y)
            mask = np.zeros_like(cut_ir_image[:,:,0])
            points = np.int32(points_warped).reshape(-1,1,2)
            mask = cv2.fillPoly(mask, [points], (255,255,255))
        cv2.imwrite("Mask.jpg",mask)
        cv2.namedWindow("MASK",cv2.WINDOW_NORMAL)
        cv2.imshow("MASK",mask)
        cv2.waitKey(0)

        cut_ir_image = cv2.applyColorMap(cut_ir_image , cv2.COLORMAP_INFERNO)
        cv2.imwrite("Cut_IR_IAMGE.jpg",cut_ir_image)
        cv2.namedWindow("Cut_IR_Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Cut_IR_Image", cut_ir_image)

        overlayed_location = (min_x, min_y, max_x - min_x, max_y - min_y)
        print("Overlayed Location:",overlayed_location)

        Cut_Visible_Image = visible_image[min_y:max_y, min_x:max_x]
        print("Cut_Visible_Image",Cut_Visible_Image.shape)
        cv2.imwrite("Cut_Visible_Image.jpg",Cut_Visible_Image)
        cv2.namedWindow("Cut_Visible_Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Cut_Visible_Image", Cut_Visible_Image)        

        registered_image = cv2.addWeighted(visible_image, 0.7, warpImage, 0.3, 0)
        cv2.imwrite("Registered_image.jpg",registered_image)
        cv2.namedWindow("Registered Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Registered Image",registered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return visible_image,ir_image,cut_ir_image,Cut_Visible_Image,overlayed_location,mask