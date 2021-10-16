from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QPushButton
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from process_image import *
from meshing import *
import os
import time
import open3d as o3d
import random
import math
import copy
import time
import shutil
import cv2 as cv


def get_rotation_matrix(angles):

    if len(angles) != 3:
        print("Only three rotation angles should be given to the function get_rotation_matrix")
        return

    alpha = (angles[0]/180) * math.pi
    beta = (angles[1]/180) * math.pi
    gamma = (angles[2]/180) * math.pi

    rotation_x = np.zeros((3,3))
    rotation_x[0][0] = 1.0
    rotation_x[1][1] = math.cos(alpha)
    rotation_x[1][2] = -math.sin(alpha)
    rotation_x[2][1] = math.sin(alpha)
    rotation_x[2][2] = math.cos(alpha)

    rotation_y = np.zeros((3,3))
    rotation_y[1][1] = 1.0
    rotation_y[0][0] = math.cos(beta)
    rotation_y[0][2] = math.sin(beta)
    rotation_y[2][0] = -math.sin(beta)
    rotation_y[2][2] = math.cos(beta)

    rotation_z = np.zeros((3,3))
    rotation_z[2][2] = 1.0
    rotation_z[0][0] = math.cos(gamma)
    rotation_z[0][1] = -math.sin(gamma)
    rotation_z[1][0] = math.sin(gamma)
    rotation_z[1][1] = math.cos(gamma)

    rotation_matrix = rotation_z.dot(rotation_y.dot(rotation_x))

    return rotation_matrix

def calc_xy_area(mesh, rotation= (0, 0, 0)):

    temp_mesh = copy.deepcopy(mesh).rotate(get_rotation_matrix(rotation))
    Max_bound = temp_mesh.get_max_bound()
    Min_bound = temp_mesh.get_min_bound()
    area = abs((Max_bound[0]-Min_bound[0]) * (Max_bound[1]-Min_bound[1]))

    return area

def calc_derivative(mesh, rotation, step, all=True):

    area_0 = calc_xy_area(mesh, rotation= rotation)

    if all:

        rotation_1 = (rotation[0]+step, rotation[1], rotation[2])
        area_1 = calc_xy_area(mesh, rotation= rotation_1)
        deriv_1 = (area_1-area_0)/step
        #print("After Rotation:" + str(rotation_1) + " dev = " + str(deriv_1))

        rotation_2 = (rotation[0], rotation[1]+step, rotation[2])
        area_2 = calc_xy_area(mesh, rotation= rotation_2)
        deriv_2 = (area_2-area_0)/step
        #print("After Rotation:" + str(rotation_2) + " dev = " + str(deriv_2))

        rotation_3 = (rotation[0], rotation[1], rotation[2]+step)
        area_3 = calc_xy_area(mesh, rotation= rotation_3)
        deriv_3 = (area_3-area_0)/step
        #print("After Rotation:" + str(rotation_3) + " dev = " + str(deriv_3))

        #print("Calculating Derivateves after:" + str((rotation_1, rotation_2, rotation_3)))

        return (deriv_1, deriv_2, deriv_3)

    else:
        rotation_3 = (rotation[0], rotation[1], rotation[2]+step)
        area_3 = calc_xy_area(mesh, rotation= rotation_3)
        deriv_3 = (area_3-area_0)/step

        return deriv_3



def fix_mesh_orientation(mesh, max_iteration = -1, step= 0.5, acceptable_error= 0.00001):

    temp_mesh = copy.deepcopy(mesh)

    center = temp_mesh.get_center()
    temp_mesh = temp_mesh.translate(-center)

    Max_bound = temp_mesh.get_max_bound()
    Min_bound = temp_mesh.get_min_bound()
    max_area = calc_xy_area(temp_mesh)

    rotation_angles = (0, 0, 0)

    temp_rotation = (0, 0, 0)

    i = 1

    while True:

        derivative = calc_derivative(temp_mesh, temp_rotation, step)
        temp_rotation = (temp_rotation[0] + step*derivative[0], temp_rotation[1] + step*derivative[1], temp_rotation[2] + step*derivative[2])
        xy_area = calc_xy_area(temp_mesh, temp_rotation)

        print("Iteration " + str(i) + " >>>>>> Area: " + str(xy_area))

        if xy_area > max_area + max_area*acceptable_error:
            max_area = xy_area
            rotation_angles = temp_rotation
        
        else:
            break
        
        i += 1

        if i == max_iteration:
            break
        
    
    temp_mesh = temp_mesh.rotate(get_rotation_matrix(rotation_angles))


    return temp_mesh, max_area, rotation_angles
        

def flip_if_needed(mesh):
    Max_bound = mesh.get_max_bound()
    Min_bound = mesh.get_min_bound()

    min_b = (Min_bound[0]*0.8, Min_bound[1]*0.8, Min_bound[2])
    max_b = (Max_bound[0]*0.8, Max_bound[1]*0.8, Max_bound[2])

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)

    croped_mesh = copy.deepcopy(mesh).crop(bbox)

    center = croped_mesh.get_center()

    if center[2] < 0:
        mesh = mesh.rotate(get_rotation_matrix((180, 0, 0)))

    return mesh


def get_object_xy_limits(mesh):

    center = mesh.get_center()
    x_0 = center[0]
    y_0 = center[1]
    z_0 = center[2]
    iss = []
    xmin_xs = []
    xmin_ys = []
    xmin_zs = []
    xmin_dz = []
    xmin_last_z = 0

    xmax_xs = []
    xmax_ys = []
    xmax_zs = []
    xmax_dz = []
    xmax_last_z = 0

    ymin_xs = []
    ymin_ys = []
    ymin_zs = []
    ymin_dz = []
    ymin_last_z = 0

    ymax_xs = []
    ymax_ys = []
    ymax_zs = []
    ymax_dz = []
    xmax_last_z = 0

    Max_bound = mesh.get_max_bound()
    Min_bound = mesh.get_min_bound()


    for i in range(1,100):
        #### >>>>>>>>>>>>>>>>>>   Find minimum x value for the object
        min_b = (x_0 - (abs(x_0 - Min_bound[0]) * (i / 100)), Min_bound[1], Min_bound[2])
        max_b = (Max_bound[0], Max_bound[1], Max_bound[2])

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)

        croped_mesh = copy.deepcopy(mesh).crop(bbox)

        center = croped_mesh.get_center()

        iss.append(i/100.0)
        xmin_xs.append(center[0])
        xmin_ys.append(center[1])
        xmin_zs.append(center[2])

        if i != 1:
            dz = center[2] - xmin_last_z
            xmin_last_z = center[2]
            xmin_dz.append(dz)
        else:
            xmin_last_z = center[2]

        #### >>>>>>>>>>>>>>>>>>   Find maximum x value for the object
        min_b = (Min_bound[0], Min_bound[1], Min_bound[2])
        max_b = (x_0 + (abs(Max_bound[0] - x_0) * (i / 100)), Max_bound[1], Max_bound[2])

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)

        croped_mesh = copy.deepcopy(mesh).crop(bbox)

        center = croped_mesh.get_center()

        xmax_xs.append(center[0])
        xmax_ys.append(center[1])
        xmax_zs.append(center[2])

        if i != 1:
            dz = center[2] - xmax_last_z
            xmax_last_z = center[2]
            xmax_dz.append(dz)
        else:
            xmax_last_z = center[2]

        #### >>>>>>>>>>>>>>>>>>   Find minimum y value for the object
        min_b = (Min_bound[0], y_0 - (abs(y_0 - Min_bound[1]) * (i / 100)), Min_bound[2])
        max_b = (Max_bound[0], Max_bound[1], Max_bound[2])

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)

        croped_mesh = copy.deepcopy(mesh).crop(bbox)

        center = croped_mesh.get_center()

        ymin_xs.append(center[0])
        ymin_ys.append(center[1])
        ymin_zs.append(center[2])

        if i != 1:
            dz = center[2] - ymin_last_z
            ymin_last_z = center[2]
            ymin_dz.append(dz)
        else:
            ymin_last_z = center[2]

        #### >>>>>>>>>>>>>>>>>>   Find maximum y value for the object
        min_b = (Min_bound[0], Min_bound[1], Min_bound[2])
        max_b = (Max_bound[0], y_0 + (abs(Max_bound[1] - y_0) * (i / 100)), Max_bound[2])

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)

        croped_mesh = copy.deepcopy(mesh).crop(bbox)

        center = croped_mesh.get_center()

        ymax_xs.append(center[0])
        ymax_ys.append(center[1])
        ymax_zs.append(center[2])

        if i != 1:
            dz = center[2] - ymax_last_z
            ymax_last_z = center[2]
            ymax_dz.append(dz)
        else:
            ymax_last_z = center[2]

        print(str(i))
    
    x_min_limit = (xmin_dz.index(min(xmin_dz)) + 12) / 100.0
    x_max_limit = (xmax_dz.index(min(xmax_dz)) + 12) / 100.0

    y_min_limit = (ymin_dz.index(min(ymin_dz)) + 12) / 100.0
    y_max_limit = (ymax_dz.index(min(ymax_dz)) + 12) / 100.0
    
    return (x_min_limit, x_max_limit, y_min_limit, y_max_limit)

def fix_obj(path):

    mesh = o3d.io.read_triangle_mesh(path)

    TIC = time.time()

    tic = time.time()
    fixed_mesh, max_area, rotation_angles = fix_mesh_orientation(mesh, max_iteration = -1, step= 2, acceptable_error= 0.001)
    toc = time.time()

    Max_bound = fixed_mesh.get_max_bound()
    Min_bound = fixed_mesh.get_min_bound()

    min_b = (Min_bound[0]*0.7, Min_bound[1]*0.7, Min_bound[2])
    max_b = (Max_bound[0]*0.7, Max_bound[1]*0.7, Max_bound[2])

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)

    croped_mesh = copy.deepcopy(fixed_mesh).crop(bbox)

    tic = time.time()
    fixed_mesh, max_area, rotation_angles = fix_mesh_orientation(croped_mesh, max_iteration = -1, step= 2, acceptable_error= 0.000001)
    toc = time.time()

    fixed_mesh = flip_if_needed(fixed_mesh)

    print(toc-tic)

    limits_1 = get_object_xy_limits(fixed_mesh)

    center = fixed_mesh.get_center()

    last_zs = center[2]
    highest_derivative = 0
    cut_off_i = 0

    for i in range(1,100):

        Max_bound = fixed_mesh.get_max_bound()
        Min_bound = fixed_mesh.get_min_bound()

        min_b = (Min_bound[0], Min_bound[1], Min_bound[2] + (Max_bound[2] - Min_bound[2])*(i/100.0) )
        max_b = (Max_bound[0], Max_bound[1], Max_bound[2])

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)

        croped_mesh = copy.deepcopy(fixed_mesh).crop(bbox)

        center = croped_mesh.get_center()

        dz = center[2] - last_zs
        if dz > highest_derivative:
            print(cut_off_i)
            highest_derivative = dz
            cut_off_i = i
        
        last_zs = center[2]

        print(str(i) + " >>>>> " + str(center))

    Max_bound = fixed_mesh.get_max_bound()
    Min_bound = fixed_mesh.get_min_bound()

    min_b = (Min_bound[0], Min_bound[1], Min_bound[2] + (Max_bound[2] - Min_bound[2])*((cut_off_i)/100.0) )
    max_b = (Max_bound[0], Max_bound[1], Max_bound[2])

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)

    croped_mesh = copy.deepcopy(fixed_mesh).crop(bbox)

    Max_bound_t = croped_mesh.get_max_bound()
    Min_bound_t = croped_mesh.get_min_bound()

    min_b = (limits_1[0] * Min_bound[0], limits_1[2] * Min_bound[1], Min_bound_t[2])
    max_b = (limits_1[1] * Max_bound[0], limits_1[3] * Max_bound[1], Max_bound_t[2])

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)

    croped_mesh = copy.deepcopy(croped_mesh).crop(bbox)

    center = fixed_mesh.get_center()
    Max_bound = fixed_mesh.get_max_bound()
    Min_bound = fixed_mesh.get_min_bound()

    print(center)
    print(Max_bound)
    print(Min_bound)

    o3d.io.write_triangle_mesh(path.replace("filteredmesh", "cropped_mesh"), croped_mesh)

    TOC = time.time()

    print("The Whole Process took " + str(TOC-TIC) + " Seconds")

    #o3d.visualization.draw_geometries([croped_mesh])

def run_command_line(cmd):
    os.system(cmd)

def mesh(images_pathes, processed_images_pathes, progress_label, progress_bar, aliceVision_path, app, use_sift, use_akaze, speed_up=1):

    descriptors = ""
    message = ""
    if use_sift and use_akaze:
        descriptors += "sift,akaze"
        message = "SIFT and AKAZE"
    elif use_sift:
        descriptors = "sift"
        message = "SIFT"
    else:
        descriptors = "akaze"
        message = "AKAZE"

    if not speed_up:
        speed_up = 1

    times = []
    
    file_dir = os.path.dirname(__file__)

    _translate = QtCore.QCoreApplication.translate
    progress_bar.setValue(0)
    progress_label.setText(_translate("MainWindow", "Creating directory for run time data ..."))
    time.sleep(1)

    ### >>>>>>>>>>> Creating RunTimeDataDirectory
    dir_path = os.path.join(file_dir, "RunTimeData")
    try:
        os.mkdir(dir_path)
    except:
        os.system("rmdir " + dir_path + " /s /q")
        os.mkdir(dir_path)
    
    ### >>>>>>>>>>> Copying original images
    
    progress_label.setText(_translate("MainWindow", "Copying all original images to one directory ..."))
    time.sleep(1)

    images_path = os.path.join(dir_path, "original_images")
    os.mkdir(images_path)

    for path in images_pathes:
        file_name = path.split('/')
        file_name = file_name[len(file_name) - 1]
        temp_dir = images_path + "\\" + file_name
        shutil.copyfile(path, temp_dir)
        

    ### >>>>>>>>>>> Copying processed images
    progress_label.setText(_translate("MainWindow", "Copying and preparing all processed images to one directory ..."))
    app.processEvents()

    processed_images_path = os.path.join(dir_path, "processed_images")
    os.mkdir(processed_images_path)

    for path in processed_images_pathes:
        file_name = path.split('/')
        file_name = file_name[len(file_name) - 1]
        temp_dir = processed_images_path + "\\" + file_name
        shutil.copyfile(path, temp_dir)
    
    if speed_up != 1:

        images_list = os.listdir(processed_images_path)

        for img in images_list:
            path = processed_images_path + "/" + img
            img = cv.imread(path)
            size_1 = int(img.shape[0]/(speed_up**0.5))
            size_2 = int(img.shape[1]/(speed_up**0.5))
            img = cv.resize(img, (size_2, size_1))
            cv.imwrite(path, img)

    tic = time.time()


    ### >>>>>>>> Computing the intrinsics

    progress_bar.setValue(1)
    progress_label.setText(_translate("MainWindow", "Calculating the intrinsic camera matrix for all input images ..."))
    app.processEvents()
    time.sleep(1)

    intrinsics_directory = os.path.join(dir_path, "camera_information")

    os.mkdir(intrinsics_directory)

    print("Calculating the intirinsic camera matrix for all images")

    camera_matrix_info_path = intrinsics_directory + "/cameraInit.sfm"
    
    os.system(aliceVision_path + "\\bin\\aliceVision_cameraInit.exe --defaultFieldOfView 45.0 --verboseLevel info --sensorDatabase \"\" --allowSingleView 1 --imageFolder \"" + images_path + "\"  --output \"" + camera_matrix_info_path + "\"")

    if os.path.isfile(camera_matrix_info_path):
        ### >>>>>>>> Changing files pathes in cameraInit.sfm
        camera_matrix_info = open(camera_matrix_info_path, "r")

        new_camera_matrix_info = ""

        camera_matrix_info = camera_matrix_info.readlines(-1)

        jump = 0

        for i in range(len(camera_matrix_info)):

            line = camera_matrix_info[i]

            if jump == 0:
                if line.find('"path"') != -1:
                    #print("I found path")
                    old_file_name = line.replace("\"path: \"", "").split('\\')
                    old_file_name = old_file_name[len(old_file_name)-1].replace("\",", "").replace("\n", "").strip()

                    line = "\"path\": \"" + processed_images_path + "\\" + old_file_name + "\",\n"
                    line = line.replace("\\", "\\\\")
                
                elif line.find('"Exif:PixelXDimension"') != -1 and speed_up != 1:
                    #print("I found Exif:PixelXDimension")
                    old_dimension = int(line.replace('"Exif:PixelXDimension": "', "").replace('",', "").replace("\n", ""))
                    new_dimension = int(old_dimension/(speed_up**0.5))
                    line = line.replace(str(old_dimension), str(new_dimension))

                elif line.find('"width"') != -1 and speed_up != 1:
                    #print("I found width")
                    old_dimension = int(line.replace('"width": "', "").replace('",', "").replace("\n", ""))
                    new_dimension = int(old_dimension/(speed_up**0.5))
                    line = line.replace(str(old_dimension), str(new_dimension))

                elif line.find('"pxInitialFocalLength"') != -1 and speed_up != 1:
                    #print("I found pxInitialFocalLength")
                    old_dimension = float(line.replace('"pxInitialFocalLength": "', "").replace('",', "").replace("\n", ""))
                    new_dimension = float(old_dimension/(speed_up**0.5))
                    line = line.replace(str(old_dimension), str(new_dimension))

                elif line.find('"pxFocalLength"') != -1 and speed_up != 1:
                    #print("I found pxFocalLength")
                    old_dimension = float(line.replace('"pxFocalLength": "', "").replace('",', "").replace("\n", ""))
                    new_dimension = float(old_dimension/(speed_up**0.5))
                    line = line.replace(str(old_dimension), str(new_dimension))

                elif line.find('"Exif:PixelYDimension"') != -1 and speed_up != 1:
                    #print("I found Exif:PixelYDimension")
                    old_dimension = int(line.replace('"Exif:PixelYDimension": "', "").replace('",', "").replace("\n", ""))
                    new_dimension = int(old_dimension/(speed_up**0.5))
                    line = line.replace(str(old_dimension), str(new_dimension))
                
                elif line.find('"height"') != -1 and speed_up != 1:
                    #print("I found height")
                    old_dimension = int(line.replace('"height": "', "").replace('",', "").replace("\n", ""))
                    new_dimension = int(old_dimension/(speed_up**0.5))
                    line = line.replace(str(old_dimension), str(new_dimension))
                
                elif line.find('"XResolution"') != -1 and speed_up != 1:
                    #print("I found height")
                    old_dimension = int(line.replace('"XResolution": "', "").replace('",', "").replace("\n", ""))
                    new_dimension = int(old_dimension/(speed_up**0.5))
                    line = line.replace(str(old_dimension), str(new_dimension))

                elif line.find('"YResolution"') != -1 and speed_up != 1:
                    #print("I found height")
                    old_dimension = int(line.replace('"YResolution": "', "").replace('",', "").replace("\n", ""))
                    new_dimension = int(old_dimension/(speed_up**0.5))
                    line = line.replace(str(old_dimension), str(new_dimension))

                elif line.find('"principalPoint"') != -1 and speed_up != 1:
                    #print("I found principalPoint")
                    next_line = camera_matrix_info[i+1]
                    nexter_line = camera_matrix_info[i+2]

                    old_dimension = int(next_line.replace('"', "").replace(',', "").replace("\n", ""))
                    new_dimension = int(old_dimension/(speed_up**0.5))
                    next_line = next_line.replace(str(old_dimension), str(new_dimension))

                    old_dimension = int(nexter_line.replace('"', "").replace("'", "").replace(',', "").replace("\n", ""))
                    new_dimension = int(old_dimension/(speed_up**0.5))
                    nexter_line = nexter_line.replace(str(old_dimension), str(new_dimension))

                    line = line + next_line + nexter_line + camera_matrix_info[i+3]
                    jump = 3
                
                new_camera_matrix_info += line

            else:
                jump -= 1


            

        f = open(camera_matrix_info_path, "w")
        f.write(new_camera_matrix_info)
        f.close

        print("Intirinsic Camera Matrix was calculated successfully!")

        toc = time.time()
        times.append(toc - tic)

        ### >>>>>>>>>>>> Extracting Features
        progress_label.setText(_translate("MainWindow", "Extracting features using " + message + " ..."))
        app.processEvents()
        time.sleep(1)

        print("Extracting features")

        features = os.path.join(dir_path, "features")

        os.mkdir(features)

        os.system(aliceVision_path + "\\bin\\aliceVision_featureExtraction.exe --describerTypes " + descriptors + " --describerPreset ultra --describerQuality ultra --contrastFiltering GridSort --gridFiltering True --forceCpuExtraction False --maxThreads 0 --verboseLevel info  --rangeStart 0 --rangeSize " + str(len(images_pathes)) + " --input \"" + camera_matrix_info_path + "\" --output \"" + features + "\"" )

        print("Extracting features was successful")

        toc = time.time()

        times.append(toc - tic)

        percentage = 20 - (len(images_pathes)**2)*0.0055

        progress_bar.setValue(int(percentage))
        progress_label.setText(_translate("MainWindow", "Matching images ..."))
        app.processEvents()
        time.sleep(1)

        ### >>>>>>>>>>> Matching Images

        print("Matching Images Started")

        images_matches_path = os.path.join(dir_path, "image_matches")
        os.mkdir(images_matches_path)
        images_matches = images_matches_path + "/matches.txt"

        os.system(aliceVision_path + "\\bin\\aliceVision_imageMatching.exe --minNbImages 200 --method SequentialAndVocabularyTree  --tree "" --maxDescriptors 500 --verboseLevel info --weights "" --nbMatches 50 --input \"" + camera_matrix_info_path + "\" --featuresFolder \"" + features + "\" --output \"" + images_matches + "\"")

        print("Matching Images was done successfully")

        toc = time.time()

        times.append(toc - tic)

        ### >>>>>>>>> Matching Features

        progress_label.setText(_translate("MainWindow", "Matching features ..."))
        app.processEvents()
        time.sleep(1)

        print("Matching Features Started")

        feature_matches = os.path.join(dir_path, "feature_matches")

        os.mkdir(feature_matches)

        os.system(aliceVision_path + "\\bin\\aliceVision_featureMatching.exe  --verboseLevel info --describerTypes " + descriptors + " --maxMatches 0 --exportDebugFiles False --savePutativeMatches False --guidedMatching False  --geometricEstimator acransac --geometricFilterType fundamental_matrix --maxIteration 2048 --distanceRatio 0.8  --photometricMatchingMethod ANN_L2  --imagePairsList \"" + images_matches + "\" --input \"" + camera_matrix_info_path + "\"  --featuresFolders \"" + features + "\" --output \"" + feature_matches + "\"")

        print("Matching Features was done successfully")

        toc = time.time()

        times.append(toc - tic)

        ### >>>>>>>>>>> Structure from Motion

        percentage = 70 - ((len(images_pathes) - 200)**2)*0.0004

        progress_bar.setValue(int(percentage))
        progress_label.setText(_translate("MainWindow", "Constructing structure from motion ..."))
        app.processEvents()
        time.sleep(1)

        print("Constructing Structure from motion started")

        poses_info = os.path.join(dir_path, "poses_info")
        extra_info = os.path.join(dir_path, "extra_info")
        structure_from_motion = os.path.join(dir_path, "structure_from_motion")

        os.mkdir(poses_info)
        os.mkdir(extra_info)
        os.mkdir(structure_from_motion)

        os.system(aliceVision_path + "\\bin\\aliceVision_incrementalSfm.exe --minAngleForLandmark 2.0 --minNumberOfObservationsForTriangulation 2 --maxAngleInitialPair 40.0 --maxNumberOfMatches 0 --localizerEstimator acransac --describerTypes " + descriptors + " --lockScenePreviouslyReconstructed False --localBAGraphDistance 1 --interFileExtension .ply --useLocalBA True  --minInputTrackLength 2 --useOnlyMatchesFromInputFolder False --verboseLevel info --minAngleForTriangulation 3.0 --maxReprojectionError 4.0 --minAngleInitialPair 5.0  --input \"" + camera_matrix_info_path + "\"  --featuresFolders \"" + features + "\"  --matchesFolders \"" + feature_matches + "\"  --outputViewsAndPoses \"" + poses_info + "/cameras.sfm\"  --extraInfoFolder \"" + extra_info + "\" --output \"" + structure_from_motion + "/bundle.sfm\"")

        print("Constructing Structure from motion was successful")

        toc = time.time()

        times.append(toc - tic)


        ### >>>>>>>>>> Preparing Dense Scene

        progress_bar.setValue(int(percentage) + 10)
        progress_label.setText(_translate("MainWindow", "Preparing dense scene ..."))
        app.processEvents()
        time.sleep(1)

        print("Preparing Dense Scene")

        dense_scene = os.path.join(dir_path, "prepare_dense_scene")

        os.mkdir(dense_scene)

        os.system(aliceVision_path + "\\bin\\aliceVision_prepareDenseScene.exe  --verboseLevel info  --input \"" + structure_from_motion + "\\bundle.sfm\" --output \"" + dense_scene + "\"")

        print("Dense Scene has been prepared successfully")

        toc = time.time()

        times.append(toc - tic)

        ### >>>>>>>>>> Creating Depth Map

        progress_bar.setValue(int(percentage) + 12)
        progress_label.setText(_translate("MainWindow", "Estimating depth map ..."))
        app.processEvents()
        time.sleep(1)

        print("Creating Depth Map")

        depth_map = os.path.join(dir_path, "depth_map")

        os.mkdir(depth_map)

        ### >>>>>> Renaming images with their viewId in cameraInit.sfm
        camera_matrix_info = open(camera_matrix_info_path, "r")

        temp = ""

        for line in camera_matrix_info:
            if line.find('"viewId"') != -1:
                temp = (line.replace("\"viewId\": ", "")).replace(',', "").replace("\n", "").replace("\"", "").strip()
            
            if line.find('"path"') != -1:
                old_file_name = line.replace("\"path: \"", "").split('\\')
                old_file_name = old_file_name[len(old_file_name)-1].replace("\",", "").replace("\n", "").strip()

                os.rename(processed_images_path.replace("\\\\", "\\") + "\\" + old_file_name, processed_images_path.replace("\\\\", "\\")  + "\\" + temp + ".png")

        camera_matrix_info.close

        os.system(aliceVision_path + "\\bin\\aliceVision_depthMapEstimation.exe --sgmGammaC 5.5 --sgmWSH 4 --refineGammaP 8.0 --refineSigma 15 --refineNSamplesHalf 150 --sgmMaxTCams 10 --refineWSH 3 --downscale 2 --refineMaxTCams 6 --verboseLevel info --refineGammaC 15.5 --sgmGammaP 8.0  --refineNiters 100 --refineNDepthsToRefine 31 --refineUseTcOrRcPixSize False  --input \"" + structure_from_motion + "\\bundle.sfm\" --imagesFolder \"" + processed_images_path + "\" --output \"" + depth_map + "\"")

        print("Depth Map Estimation was done successfully")

        toc = time.time()

        times.append(toc - tic)


        ### >>>>>>>> Applying Depth Map Filter

        progress_bar.setValue(83)
        progress_label.setText(_translate("MainWindow", "Filtering depth map ..."))
        app.processEvents()
        time.sleep(1)

        print("Applying Depth Map Filter")

        depth_map_filter = os.path.join(dir_path, "depth_map_filter")

        os.mkdir(depth_map_filter)

        os.system(aliceVision_path + "\\bin\\aliceVision_depthMapFiltering.exe  --minNumOfConsistentCamsWithLowSimilarity 4  --minNumOfConsistentCams 3 --verboseLevel info --pixSizeBall 0 --pixSizeBallWithLowSimilarity 0 --nNearestCams 10 --input \"" + structure_from_motion + "\\bundle.sfm\"  --depthMapsFolder \"" + depth_map + "\" --output \"" + depth_map_filter + "\"")

        print("Depth Map Filter was applied")

        toc = time.time()

        times.append(toc - tic)

        ### >>>>>>>>>>>> Meshing
        progress_bar.setValue(85)
        progress_label.setText(_translate("MainWindow", "Starting meshing process ..."))
        app.processEvents()
        time.sleep(1)

        print("Start of Meshing Process")

        mesh_path = os.path.join(dir_path, "meshing")

        os.mkdir(mesh_path)

        os.system(aliceVision_path + "\\bin\\aliceVision_meshing.exe  --simGaussianSizeInit 10.0 --maxInputPoints 50000000 --repartition multiResolution  --simGaussianSize 10.0 --simFactor 15.0 --voteMarginFactor 4.0 --contributeMarginFactor 2.0 --minStep 2 --pixSizeMarginFinalCoef 4.0 --maxPoints 5000000 --maxPointsPerVoxel 1000000 --angleFactor 15.0 --partitioning singleBlock --minAngleThreshold 1.0 --pixSizeMarginInitCoef 2.0 --refineFuse True --verboseLevel info --input \"" + structure_from_motion + "\\bundle.sfm\" --depthMapsFolder \"" + depth_map_filter + "\" --output \"" + mesh_path + "/mesh.sfm\" --outputMesh \"" + mesh_path + "/mesh.obj\"")

        print("Meshing is done")

        toc = time.time()

        times.append(toc - tic)

        ### >>>>>>>>>>>> Filtering Mesh
        if os.path.isfile(mesh_path + "\\mesh.obj"):
            progress_bar.setValue(87)
            progress_label.setText(_translate("MainWindow", "Filtering and smoothing mesh ..."))
            app.processEvents()
            time.sleep(1)

            print("Filtering Mesh")

            os.system(aliceVision_path + "\\bin\\aliceVision_meshFiltering.exe --smoothingIterations 10 --smoothingLambda 1 --filteringSubset all --filteringIterations 5 --filterLargeTrianglesFactor 90 --inputMesh \"" + mesh_path + "/mesh.obj\" --outputMesh \"" + mesh_path + "/filteredmesh.obj\"")
            for i in range(9, 1, -1):
                lamda = i/10.0
                os.system(aliceVision_path + "\\bin\\aliceVision_meshFiltering.exe --smoothingIterations 10 --smoothingLambda " + str(lamda) + " --filteringSubset all --filteringIterations 5 --filterLargeTrianglesFactor 90 --inputMesh \"" + mesh_path + "/filteredmesh.obj\" --outputMesh \"" + mesh_path + "/filteredmesh.obj\"")

            for i in range(90, 1, -5):
                lamda = i/1000.0
                os.system(aliceVision_path + "\\bin\\aliceVision_meshFiltering.exe --smoothingIterations 10 --smoothingLambda " + str(lamda) + " --filteringSubset all --filteringIterations 5 --filterLargeTrianglesFactor 90 --inputMesh \"" + mesh_path + "/filteredmesh.obj\" --outputMesh \"" + mesh_path + "/filteredmesh.obj\"")

            progress_bar.setValue(90)
            progress_label.setText(_translate("MainWindow", "Correcting mesh orientation and cropping object from mesh ..."))
            time.sleep(1)

            fix_obj(mesh_path + "/filteredmesh.obj")

            ### >>>>>>>>>>>> Converting .obj to .stl

            progress_bar.setValue(99)
            progress_label.setText(_translate("MainWindow", "Cenverting original mesh.obj to mesh.stl ..."))
            app.processEvents()

            print("Converting the Mesh from .Obj to .STL")

            os.system(aliceVision_path + "\\bin\\aliceVision_convertMesh.exe  --inputMesh \"" + mesh_path + "\\mesh.obj\" --output \"" + mesh_path + "/mesh.stl\"")

            print("Converting the mesh is done")

            progress_label.setText(_translate("MainWindow", "Cenverting filteredmesh.obj to filteredmesh.stl ..."))

            print("Converting the Mesh from .Obj to .STL")

            os.system(aliceVision_path + "\\bin\\aliceVision_convertMesh.exe  --inputMesh \"" + mesh_path + "\\filteredmesh.obj\" --output \"" + mesh_path + "/filteredmesh.stl\"")

            print("Converting the mesh is done")

            progress_label.setText(_translate("MainWindow", "Cenverting cropped_mesh.obj to cropped_mesh.stl ..."))

            print("Converting the Mesh from .Obj to .STL")

            os.system(aliceVision_path + "\\bin\\aliceVision_convertMesh.exe  --inputMesh \"" + mesh_path + "\\cropped_mesh.obj\" --output \"" + mesh_path + "/cropped_mesh.stl\"")

            print("Converting the mesh is done")

            times.append(toc - tic)
        
            progress_bar.setValue(100)
            progress_label.setText(_translate("MainWindow", "Successfully completed"))
        
        else:
            progress_bar.setValue(0)
            progress_label.setText(_translate("MainWindow", "Something went wrong!! Meshing failed!"))

    else:
        progress_bar.setValue(0)
        progress_label.setText(_translate("MainWindow", "Some important camera information are missing !! Process failed!"))











