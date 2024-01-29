from holistic_edge_det import CropLayer2,CropLayer3
from edge_detection import EdgeDetector
from temp_thresholding_fusion import ThermalImageProcessing
from Image_Registration_Geometrically import CameraFusion
from image_Registration_Homographic import ImageRegistration
import numpy as np
import argparse
import sys
import math
import json
import cv2
with open('info.json', 'r') as f:
    input_params = json.load(f)

visible_image_path = input_params['visible_image_path']
ir_image_path = input_params['ir_image_path']
camera_params = input_params['camera_params']
threshold_params = input_params["threshold"]

image_Registration_bool = input_params["image_Regis"]
detection = input_params["Detection"]

parser = argparse.ArgumentParser(description='Run image processing scripts')
subparsers = parser.add_subparsers(dest='script', help='sub-command help')


#image_Registration
image_reg_parser = subparsers.add_parser('image_Registration_Homographic', help='image registration help')
image_reg_subparsers = image_reg_parser.add_subparsers(dest='sub_script', help='sub-command help')
holistic_edge_det_parser = image_reg_subparsers.add_parser('holistic_edge_det', help='holistic edge detection help')
edge_det_parser = image_reg_subparsers.add_parser('edge_detection', help='edge detection help')
temp_thresh_fusion_parser = image_reg_subparsers.add_parser('temp_thresholding_fusion', help='temporal thresholding fusion help')


#image_Registration_Geometrically
image_reg_geo_parser = subparsers.add_parser('image_Registration_Geometrically', help='image registration geometrically help')
image_reg_geo_subparsers = image_reg_geo_parser.add_subparsers(dest='sub_script', help='sub-command help')
holistic_edge_det_parser = image_reg_geo_subparsers.add_parser('holistic_edge_det', help='holistic edge detection help')
edge_det_parser = image_reg_geo_subparsers.add_parser('edge_detection', help='edge detection help')
temp_thresh_fusion_parser = image_reg_geo_subparsers.add_parser('temp_thresholding_fusion', help='temporal thresholding fusion help')

args = parser.parse_args()

if image_Registration_bool == True:    
# if args.script == 'image_Registration':
    print("Running image_Registration_Homographic script")
    
    if detection == 'holistic_edge_det':
        proto_path = input_params['proto_path']
        model_path = input_params['model_path']
        IR_image = cv2.imread(ir_image_path)
        Visible_image = cv2.imread(visible_image_path)
        center = (IR_image.shape[1] // 2, IR_image.shape[0] // 2)
        angle = 45
        scale = 1
        rotation_matrix = cv2.getRotationMatrix2D(center,angle,scale)# returns a 2*3 matrix
        homography_matrix = np.vstack([rotation_matrix, [0, 0, 1]])# making 2*3 matrix to 3*3 matrix
        print("Homography Matrix",homography_matrix)
        image_registration = ImageRegistration(homography_matrix)
        Visible_image,ir_image,cut_ir_image,Cut_Visible_Image,overlayed_location,mask = image_registration.register_image(Visible_image,IR_image)
        homography_register_image = CropLayer3()
        homography_register_image.display_HED_Homography(proto_path,model_path,Visible_image,cut_ir_image,Cut_Visible_Image,overlayed_location,mask)

    elif detection == 'edge_detection':

        threshold = int(input_params["threshold"])
        IR_image = cv2.imread(ir_image_path)
        Visible_image = cv2.imread(visible_image_path)
        center = (IR_image.shape[1] // 2, IR_image.shape[0] // 2)
        angle = 50
        scale = 1
        rotation_matrix = cv2.getRotationMatrix2D(center,angle,scale)# returns a 2*3 matrix
        homography_matrix = np.vstack([rotation_matrix, [0, 0, 1]])# making 2*3 matrix to 3*3 matrix
        # c = np.cos(0)
        # s = np.sin(0)
        # tx = IR_image.shape[1] // 2
        # ty = IR_image.shape[0] // 2
        # homography_matrix = np.array([[c, -s, tx*(1-c)+ty*s],
        #                             [s, c, ty*(1-c)-tx*s],
        #                             [0, 0, 1]])
        
        # cx_v = Visible_image.shape[1] // 2
        # cy_v = Visible_image.shape[0] / / 2
        # cx_ir = IR_image.shape[1] // 2
        # cy_ir = IR_image.shape[0] // 2
        # dx = cx_v - cx_ir
        # dy = cy_v - cy_ir
        # homography_matrix[0, 2] += dx
        # homography_matrix[1, 2] += dy
        
        print("Homography Matrix",homography_matrix)
        image_registration = ImageRegistration(homography_matrix)
        visible_image,ir_image,cut_ir_image,Cut_Visible_Image,overlayed_location,mask = image_registration.register_image(Visible_image,IR_image)

        homography_register_image = EdgeDetector(visible_image,ir_image,Cut_Visible_Image)
        homography_register_image.displayOverlayHomography(threshold,overlayed_location,cut_ir_image,mask)

    elif detection == 'temp_thresholding_fusion':

        threshold = int(input_params["threshold"])
        IR_image = cv2.imread(ir_image_path)
        Visible_image = cv2.imread(visible_image_path)
        Visible_image_for_thresh_Fusion = cv2.imread(visible_image_path,0)
        center = (IR_image.shape[1] // 2, IR_image.shape[0] // 2)
        angle = 45
        scale = 1
        rotation_matrix = cv2.getRotationMatrix2D(center,angle,scale)# returns a 2*3 matrix
        homography_matrix = np.vstack([rotation_matrix, [0, 0, 1]])# making 2*3 matrix to 3*3 matrix
        image_registration = ImageRegistration(homography_matrix)
        visible_image,ir_image,cut_ir_image,Cut_Visible_Image,overlayed_location,mask = image_registration.register_image(Visible_image,IR_image)
        cv2.namedWindow("Cut-IR-Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Cut-IR-Image",cut_ir_image)
        cv2.waitKey(0)
        homography_register_image = ThermalImageProcessing(cut_ir_image,Cut_Visible_Image,Visible_image_for_thresh_Fusion)
        homography_register_image.display_image_homography(threshold,overlayed_location,mask)
else : 
# if args.script == 'image_Registration_Geometrically':
    print("Running image_Registration_Geometrically script")

    if detection == 'holistic_edge_det':

        proto_path = input_params['proto_path']
        model_path = input_params['model_path']
        Visible_image = cv2.imread(visible_image_path)
        ir_image = cv2.imread(ir_image_path)
        obj_cam_fusion = CameraFusion(Visible_image, ir_image)

        visible_image,resized_ir_image,cut_visible_image = obj_cam_fusion.calculate_distance(
                                            math.radians(camera_params["fov_horizontal_degrees"]), math.radians(camera_params["fov_vertical_degrees"])
                                            ,camera_params["distance_m"], math.radians(camera_params["pitch_degrees"]),camera_params["tx"]
                                            , camera_params["tz"])

        HED = CropLayer2()
        # HED.Generate_Edge(proto_path,model_path,visible_image_path,resized_ir_image,cut_visible_image)
        HED.display_HED(proto_path,model_path,visible_image,resized_ir_image,cut_visible_image)
        
    elif detection == 'edge_detection':

        print("Running Edge Detection script")
        Visible_image = cv2.imread(visible_image_path)
        ir_image = cv2.imread(ir_image_path)
        obj_cam_fusion = CameraFusion(Visible_image, ir_image)
        originalImage = cv2.imread(visible_image_path, 1)
        threshold = int(input_params["threshold"])
        visible_image, resized_ir_image,cut_visible_image = obj_cam_fusion.calculate_distance(
        math.radians(camera_params["fov_horizontal_degrees"]), math.radians(camera_params["fov_vertical_degrees"])
                                                ,camera_params["distance_m"],
                                                math.radians(camera_params["pitch_degrees"]),camera_params["tx"]
                                                , camera_params["tz"]) 
        edge_detector = EdgeDetector(originalImage, resized_ir_image,cut_visible_image)
        edge_detector.displayEdgeDetection(threshold)

    elif detection == 'temp_thresholding_fusion':

        print("Running temp_thresholding_fusion script")
        Visible_image = cv2.imread(visible_image_path)
        Visible_image_for_thresh_Fusion = cv2.imread(visible_image_path,0)
        ir_image = cv2.imread(ir_image_path)
        obj_cam_fusion = CameraFusion(Visible_image, ir_image)
        threshold = int(input_params["threshold"])
        visible_image, resized_ir_image,cut_visible_image = obj_cam_fusion.calculate_distance(
        math.radians(camera_params["fov_horizontal_degrees"]), math.radians(camera_params["fov_vertical_degrees"])
                                                    ,camera_params["distance_m"], math.radians(camera_params["pitch_degrees"]),camera_params["tx"]
                                                        , camera_params["tz"]) 
        thermal_detector = ThermalImageProcessing(resized_ir_image,cut_visible_image,Visible_image_for_thresh_Fusion)
        thermal_detector.display_images(threshold)

    else:
        print("Invalid script name")
sys.exit(1)