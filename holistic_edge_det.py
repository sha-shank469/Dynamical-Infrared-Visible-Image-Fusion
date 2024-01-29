import argparse
import cv2
import os
import numpy as np
import sys

# Define a custom CropLayer
class CropLayer:
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of
		# the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):
		
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		# compute the starting and ending crop coordinates
		
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]

class CropLayer2:

	def Generate_Edge(self,proto_path,model_path,visible_image,resized_ir_image,cut_visible_image):

		protoPath = proto_path
		modelPath = model_path
		net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

		# Register the custom CropLayer with the model
		cv2.dnn_registerLayer("Crop", CropLayer)

		self.originalImage = visible_image
		print("Visible Image Shape",self.originalImage.shape)
		cv2.namedWindow("Visible_Image____",cv2.WINDOW_NORMAL)
		cv2.imshow("Visible_Image____",self.originalImage)

		self.cut_vis_image = cut_visible_image
		self.resized_ir_image = resized_ir_image

		#Resiszing the Original Visible Image
		new_width = int(self.originalImage.shape[1] / 4)
		print("new_width",new_width)
		new_height = int(self.originalImage.shape[0]/ 4)
		print("New_Height",new_height)
		Resized_Visible_image = cv2.resize(self.originalImage, (new_width,new_height), interpolation=cv2.INTER_AREA)
		print("Input_Image_shape",Resized_Visible_image.shape)

		#Resizing the IR Visible image
		new_width_ir = int(self.resized_ir_image.shape[1] / 4)
		print("new width",new_width_ir)
		new_height_ir = int(self.resized_ir_image.shape[0]/ 4)
		print("New Height",new_height_ir)
		Resize_IR_IMAGE = cv2.resize(self.resized_ir_image, (new_width_ir,new_height_ir), interpolation=cv2.INTER_AREA)
		cv2.namedWindow("Resize_IR_IMAGE",cv2.WINDOW_NORMAL)
		cv2.imshow("Resize_IR_IMAGE", Resize_IR_IMAGE)
		print("Resized IR Image shape",Resize_IR_IMAGE.shape)

		# background = Resized_Visible_image.copy()
		start = (Resized_Visible_image.shape[0] // 2 - Resize_IR_IMAGE.shape[0] // 2, Resized_Visible_image.shape[1] // 2 - Resize_IR_IMAGE.shape[1] // 2)
		print("start",start)
		end = (Resized_Visible_image.shape[0] // 2  + Resize_IR_IMAGE.shape[0] // 2, Resized_Visible_image.shape[1] // 2 + Resize_IR_IMAGE.shape[1] // 2)
		print("end",end)
		height = end[0]-start[0] 
		width = end[1]-start[1]
		cut_visible_image = Resized_Visible_image.copy()
		cut_visible_image = cut_visible_image[start[0]:start[0]+height, start[1]:start[1]+width] 
		cv2.namedWindow("Input",cv2.WINDOW_NORMAL)
		cv2.imshow("Input", cut_visible_image)

		overlay_location = (start[0], start[1], end[0], end[1])
		print("Overlay Location",overlay_location)
		
		(H, W) = cut_visible_image.shape[:2]
		gray = cv2.cvtColor(cut_visible_image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		canny = cv2.Canny(blurred, 30, 150)
		blob = cv2.dnn.blobFromImage(cut_visible_image, scalefactor=1.0, size=(W, H),
			mean=(104.00698793, 116.66876762, 122.67891434),
			swapRB=False, crop=False)
		net.setInput(blob)
		hed = net.forward()
		hed = cv2.resize(hed[0, 0], (W, H))
		hed = (255 * hed).astype("uint8")
		kernel = np.ones((7, 7), np.uint8)
		hed_erode = cv2.erode(hed, kernel)
		thresh_hed = cv2.adaptiveThreshold(hed_erode,255,cv2.ADAPTIVE_THRESH_MEAN_C,
				    cv2.THRESH_BINARY,3,0.0)

		image_laplace = cut_visible_image.copy()

		Resize_IR_IMAGE = cv2.applyColorMap(Resize_IR_IMAGE, cv2.COLORMAP_INFERNO)

		resized_for_overlay = cv2.resize(Resize_IR_IMAGE, (image_laplace.shape[1], image_laplace.shape[0]))
		thresh_hed = cv2.bitwise_not(thresh_hed)
		thresh_hed = cv2.cvtColor(thresh_hed, cv2.COLOR_GRAY2BGR)
		cv2.namedWindow("HED",cv2.WINDOW_NORMAL)
		cv2.imshow("HED", thresh_hed)

		print("Ch1 : {}, Ch2 : {}".format(thresh_hed.shape, image_laplace.shape))
		print("resized_for_overlay",resized_for_overlay.shape)

		resized_for_overlay = cv2.bitwise_and(resized_for_overlay, thresh_hed)
		IR_OVERLAY = cv2.addWeighted(image_laplace,0.3,resized_for_overlay,0.7,0)
		cv2.namedWindow("IR_OVERLAY",cv2.WINDOW_NORMAL)
		cv2.imshow("IR_OVERLAY", IR_OVERLAY)
		cv2.waitKey(0)
		return IR_OVERLAY,Resized_Visible_image
	

	def display_Overlayed_Image(self,IR_OVERLAY,Resized_Visible_image):

		background = Resized_Visible_image.copy()
		start = (Resized_Visible_image.shape[0] // 2 - IR_OVERLAY.shape[0] // 2, 
	   													Resized_Visible_image.shape[1] // 2 - IR_OVERLAY.shape[1] // 2)
		print("start",start)
		end = (Resized_Visible_image.shape[0] // 2  + IR_OVERLAY.shape[0] // 2, 
	 													Resized_Visible_image.shape[1] // 2 + IR_OVERLAY.shape[1] // 2)
		print("end",end)
		resized_imageIR1 = cv2.resize(IR_OVERLAY, (end[1] - start[1], end[0] - start[0]))
		background[start[0]: end[0], start[1]: end[1]] = resized_imageIR1
		cv2.imwrite("HED.jpg",background)
		cv2.namedWindow("HED Image",cv2.WINDOW_NORMAL)
		cv2.imshow("HED Image",background)
		cv2.waitKey(0)
		return background

	def display_HED(self,proto_path,model_path,visible_image,resized_ir_image,cut_visible_image):
		IR_OVERLAY,Resized_Visible_image = self.Generate_Edge(proto_path,model_path,visible_image,resized_ir_image,cut_visible_image)
		self.display_Overlayed_Image(IR_OVERLAY,Resized_Visible_image)

class CropLayer3:

	def Generate_Edge(self,proto_path,model_path,Visible_image,cut_ir_image,Cut_Visible_Image,overlayed_location):
		
		self.Cut_visible_image = Cut_Visible_Image
		self.overlay_location = overlayed_location
		self.Cut_ir_image = cut_ir_image
		self.Visible_Image = Visible_image

		protoPath = proto_path
		modelPath = model_path

		net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

		# Register the custom CropLayer with the model
		cv2.dnn_registerLayer("Crop", CropLayer)

		x,y,w,h = self.overlay_location
		print("Overlay Location",x,y,w,h)
		
		(H, W) = self.Cut_visible_image.shape[:2]
		gray = cv2.cvtColor(self.Cut_visible_image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		canny = cv2.Canny(blurred, 30, 150)
		blob = cv2.dnn.blobFromImage(self.Cut_visible_image, scalefactor=1.0, size=(W, H),
			mean=(104.00698793, 116.66876762, 122.67891434),
			swapRB=False, crop=False)
		net.setInput(blob)
		hed = net.forward()
		hed = cv2.resize(hed[0, 0], (W, H))
		hed = (255 * hed).astype("uint8")
		kernel = np.ones((7, 7), np.uint8)
		hed_erode = cv2.erode(hed, kernel)
		thresh_hed = cv2.adaptiveThreshold(hed_erode,255,cv2.ADAPTIVE_THRESH_MEAN_C,
				    cv2.THRESH_BINARY,3,0.0)

		image_laplace = self.Cut_visible_image.copy()

		Resize_IR_IMAGE = cv2.applyColorMap(self.Cut_ir_image, cv2.COLORMAP_INFERNO)

		resized_for_overlay = cv2.resize(Resize_IR_IMAGE, (image_laplace.shape[1], image_laplace.shape[0]))
		thresh_hed = cv2.bitwise_not(thresh_hed)
		thresh_hed = cv2.cvtColor(thresh_hed, cv2.COLOR_GRAY2BGR)
		cv2.namedWindow("HED",cv2.WINDOW_NORMAL)
		cv2.imshow("HED", thresh_hed)

		print("Ch1 : {}, Ch2 : {}".format(thresh_hed.shape, image_laplace.shape))
		print("resized_for_overlay",resized_for_overlay.shape)

		resized_for_overlay = cv2.bitwise_and(resized_for_overlay, thresh_hed)
		IR_OVERLAY = cv2.addWeighted(image_laplace,0.3,resized_for_overlay,0.7,0)
		cv2.namedWindow("IR_OVERLAY",cv2.WINDOW_NORMAL)
		cv2.imshow("IR_OVERLAY", IR_OVERLAY)
		cv2.waitKey(0)
		return IR_OVERLAY,self.Visible_Image,self.overlay_location,cut_ir_image,Cut_Visible_Image
	

	def display_Overlayed_Homography(self,IR_OVERLAY,Visible_Image,cut_ir_image,Cut_Visible_Image,overlay_location,mask):
		# cv2.namedWindow("MASK",cv2.WINDOW_NORMAL)
		# cv2.imshow("MASK",mask)
		

		mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
		# cv2.namedWindow("MASK",cv2.WINDOW_NORMAL)
		# cv2.imshow("MASK",mask)

		bitwise_Image = cv2.bitwise_and(IR_OVERLAY,mask)
		cv2.namedWindow("bitwise_Image",cv2.WINDOW_NORMAL)
		cv2.imshow("bitwise_Image",bitwise_Image)

		# edge_and_IR_Overlay = cv2.bitwise_and(bitwise_Image,cut_ir_image)
		# cv2.namedWindow("Edge_And_IR_Overlay",cv2.WINDOW_NORMAL)
		# cv2.imshow("Edge_And_IR_Overlay",edge_and_IR_Overlay)
		# cv2.waitKey(0)

		background = Visible_Image.copy()
		x,y,w,h = overlay_location 
		print("OVERLAY-LOCATION",x,y,w,h)

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

		background = Visible_Image.copy()

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
		cv2.imwrite("HED_HOMOGRAPHY.jpg",background)
		cv2.namedWindow("BACKGROUND",cv2.WINDOW_NORMAL)
		cv2.imshow("BACKGROUND",background)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		# background[y:y+h,x:x+w] = cv2.addWeighted(background[y:y+h,x:x+w], 0.5, IR_OVERLAY, 0.5, 0.0)
		# cv2.namedWindow("BACKGROUND",cv2.WINDOW_NORMAL)
		# cv2.imshow("BACKGROUND",background)
		# cv2.waitKey(0)
		return background
	
	def display_HED_Homography(self,proto_path,model_path,Visible_image,cut_ir_image,
			    								Cut_Visible_Image,overlayed_location,mask):
		IR_OVERLAY,Visible_Image,overlay_location,cut_ir_image,Cut_Visible_Image = self.Generate_Edge(proto_path,model_path,Visible_image,
								 				cut_ir_image,Cut_Visible_Image,overlayed_location)
		self.display_Overlayed_Homography(IR_OVERLAY,Visible_Image,cut_ir_image,Cut_Visible_Image,overlay_location,mask)