import cv2
import math
import numpy as np


class CameraFusion:

   def __init__(self, Visible_image, ir_image):

      self.visible_image = Visible_image
      self.ir_image = ir_image
      self.ir_image = cv2.applyColorMap(self.ir_image, cv2.COLORMAP_INFERNO)

   def calculate_distance(self, alpha_visible, alpha_ir, d_m, beta, tx=0, tz=0.01):
      
      d = d_m * math.cos(beta)
      print("d:", d)
      d_visible = 2 * (d - tx) * math.tan(alpha_visible / 2)
      print("dVisible", d_visible)
      d_ir = d * (math.tan((alpha_ir / 2) + beta) +
                  math.tan((alpha_ir / 2) - beta))
      print("dIR", d_ir)

      height_ir, width_ir, _ = self.ir_image.shape
      print("Original Height IR:", height_ir)
      print("Original Width IR:", width_ir)
      height_visible, width_visible, _ = self.visible_image.shape
      print("Original Height Visible:", height_visible)
      print("Original width Visible:", width_visible)

      resized_ir_image = cv2.resize(self.ir_image, (int(width_ir * (d_ir / d_visible) * (
         height_visible / height_ir)), int(height_visible * (d_ir / d_visible))))
      print("Resized IR image shape",resized_ir_image.shape)

      tz_prime = d * math.tan(beta)
      print("tz_prime", tz_prime)
      y = math.ceil((height_visible / 2) * (tz - tz_prime)) // ((d - tx) * math.tan(alpha_visible / 2))
      y = int(y)
      print("Y:", y)

      background = self.visible_image.copy()
      start = ((height_visible // 2 - y) -
               resized_ir_image.shape[0] // 2, width_visible // 2 - resized_ir_image.shape[1] // 2)
      
      end = ((height_visible // 2 - y) +
            resized_ir_image.shape[0] // 2, width_visible // 2 + resized_ir_image.shape[1] // 2)
      
      resized_imageIR1 = cv2.resize(
         resized_ir_image, (end[1] - start[1], end[0] - start[0]))
      background[start[0]: end[0], start[1]: end[1]] = 0.5 * background[start[0]: end[0], start[1]: end[1]] + 0.5 * resized_imageIR1 
      # cv2.namedWindow("BACKGROUND", cv2.WINDOW_NORMAL)
      # cv2.imshow("BACKGROUND", background)

      height = end[0]-start[0] 
      width = end[1]-start[1] 

      # fused_image = background
      # cut_fused_image = fused_image[start[0]:start[0]+height, start[1]:start[1]+width]

      cut_visible_image = self.visible_image.copy()
      cut_visible_image = cut_visible_image[start[0]:start[0]+height, start[1]:start[1]+width]
      print("Cut visible image shape",cut_visible_image.shape)

      # cv2.namedWindow("cut_visible_image",cv2.WINDOW_NORMAL)
      # cv2.imshow("cut_visible_image",cut_visible_image)
      # cv2.waitKey(0)

      overlay_location = (start[0], start[1], end[0], end[1])
      print(overlay_location)
      
      return (self.visible_image, resized_ir_image,cut_visible_image)
   