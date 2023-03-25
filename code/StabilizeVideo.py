import cv2
from Utils.VideoUtils import VideoReader
from Utils.VideoUtils import VideoWriter
import argparse
import numpy as np
import os
import sys
import warnings
import time
warnings.filterwarnings("ignore")
'''
Instructions to run:
1. Please run the code as instructed in the Task06 pdf. eg:
    change directory to the anon_folder/code
    run the following command:
    $ python StabilizeVideo_Raw.py --question ? --subQuestion ? --video ?
    Also, please make sure directory ../data/--question/--video Contains the input video
2. For requirement.txt please see convincing directory
3. Video is saved in the relevant directory
4. Please allow upto 15 minuted for code to run (Especially for 2nd and 3rd question)
'''

def fast_detector(source, target, n=50):
    orb = cv2.ORB_create(n)
    source_keypoint = orb.detect(source, None)
    target_keypoint = orb.detect(target, None)
    return source_keypoint, target_keypoint

def brief_descriptor(source, source_keypoint, target, target_keypoint):
    orb = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    source_descriptor = orb.compute(source, source_keypoint)[1]
    target_descriptor = orb.compute(target, target_keypoint)[1]
    return source_descriptor, target_descriptor

def gaussian_1d(size=5, sig=1):
    ax = np.linspace(-(size-1)/2., (size-1)/2., size)
    gauss = np.exp(-0.5* np.square(ax) / np.square(sig))
    gauss = gauss/np.sum(gauss)
    return gauss


def smooth(arr, ker):
    smooth_arr = np.zeros_like(arr)
    pad_length = int(len(ker)*1.5)
    arr = np.pad(arr, [(pad_length,pad_length),(0,0)], 'reflect')
    for i in range(arr.shape[1]):
        smooth_arr[:,i] =  np.convolve(arr[:,i], ker, "same")[pad_length:-pad_length]
    return smooth_arr


def cropify(image):
    s = image.shape
    # zoom 104% and crop
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.05)
    return cv2.warpAffine(image, T, (s[1], s[0]))

def Q1_A(reader, vid):
    # Initialize:
    start = time.time()
    prev_frame =reader.getFrame(0)
    n_total = 800
    n_best = 50
    comp_motion = []

    # (a). Get Compact Motion Features
    for i in range(1, reader.nrFrames):
        current_frame = reader.getFrame(i)
        # Get keypoints and descriptor using ORB for previous and Next frame
        kp1, kp2 = fast_detector(prev_frame, current_frame, n_total)
        des1, des2 = brief_descriptor(prev_frame, kp1, current_frame, kp2)
        # Brute force matcher to find the correspondence
        brute_force_obj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        total_matches = brute_force_obj.match(des1, des2)
        best_matches = sorted(total_matches, key = lambda x:x.distance)[0:n_best]
        query_indices = [x.queryIdx for x in best_matches]
        target_indices = [x.trainIdx for x in best_matches]
        # Key-point coordinate corresponding to the indices
        kp1_best = [kp1[i] for i in query_indices]
        kp2_best = [kp2[i] for i in target_indices]
        # Generating the set to be used for finding the motion parameters
        points1 = [x.pt for x in kp1_best]
        points2 = [x.pt for x in kp2_best]
        arr_pt1 = np.array(points1)
        arr_pt2 = np.array(points2)
        arr_pt1 = arr_pt1.astype('int32')
        arr_pt2 = arr_pt2.astype('int32')
        # Get Compact Motion Parameters (Using Euclidean Transform, 3 independent parameters)
        M = cv2.estimateRigidTransform(arr_pt1, arr_pt2, fullAffine=False)
        # dx, dy
        dx = M[0,2]
        dy = M[1,2]
        # change in angle d_theta
        d_theta = np.arctan2(M[1,0],M[0,0])
        comp_motion.append([dx , dy , d_theta])

        prev_frame = current_frame

    #(b). Smoothing the parametes
    comp_motion = np.array(comp_motion)
    trajectory = np.cumsum(comp_motion, axis = 0)

    kernal = gaussian_1d(100, 1000)
    smooth_trajectory = smooth(trajectory,kernal)
    smooth_motion = comp_motion + smooth_trajectory - trajectory
    #(c) Frame Correction
    prev_frame = reader.getFrame(0)
    # write
    # Check for Path and create the path
    path = "../Output/{}_{}/{}.avi".format("1", "A", vid)
    path_check = "../Output/{}_{}".format("1", "A")
    if os.path.exists(path_check) == False:
        os.mkdir(path_check)
    # VideoWriter object
    writer = VideoWriter(path, reader.getFPS())
    # Write input and output Video Side by side for comparison
    frame_write = cv2.hconcat([prev_frame, prev_frame])
    writer.writeFrame(frame_write)
    for i in range(1, reader.nrFrames-1):
        current_frame = reader.getFrame(i)
        dx = smooth_motion[i,0]
        dy = smooth_motion[i,1]
        d_theta = smooth_motion[i,2]
        # Reconstruct M matrix:
        M = np.zeros((2, 3), np.float32)
        M[1,1] = M[0,0] = np.cos(d_theta)
        M[0,1] = -np.sin(d_theta)
        M[1,0] = np.sin(d_theta)
        M[0,2] = dx
        M[1,2] = dy
        frame_stabilized = cv2.warpAffine(current_frame, M, (reader.width, reader.height))
        # Uncomment Below Code for Cropping the black pixels at boundary
        #frame_stabilized = cropify(frame_stabilized)

        frame_write = cv2.hconcat([current_frame, frame_stabilized])
        writer.writeFrame(frame_write)
    writer.close()
    end = time.time()
    print("Time taken per frame for 1A: {:0.2f} seconds".format((end-start)/reader.nrFrames))

def Q1_B(reader, vid):
    # Method-1:
    start = time.time()
    prev_frame = reader.getFrame(0)
    comp_motion = []
    # (a). Get Compact Motion Features
    for i in range(1, reader.nrFrames):
        current_frame = reader.getFrame(i)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # Compute Dense Flow cv2.calcOpticalFlowFarneback
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        # image_loc_1: Pixel location in previous frame
        image_loc_1 = np.zeros_like(flow)
        image_loc_1[:, :, 0] = np.arange(w)
        image_loc_1[:, :, 1] = np.arange(h)[:, np.newaxis]
        # image_loc_2: Pixel Location in current Frame
        image_loc_2 = np.zeros_like(flow)
        image_loc_2[:, :, 0] = flow[:, :, 0] + np.arange(w)
        image_loc_2[:, :, 1] = flow[:, :, 1] + np.arange(h)[:, np.newaxis]
        # Flatten both the pixel location
        image_loc_1 = image_loc_1.reshape(h*w,2)
        image_loc_2 = image_loc_2.reshape(h*w,2)
        # Find Euclidiean Transform to find the translation and rotation
        M = cv2.estimateRigidTransform(image_loc_1, image_loc_2, fullAffine=False)
        # dx, dy
        dx = M[0, 2]
        dy = M[1, 2]
        # change in angle d_theta
        d_theta = np.arctan2(M[1, 0], M[0, 0])
        comp_motion.append([dx, dy, d_theta])
        prev_frame = current_frame

    # (b). Smoothing the parametes (Same as 1A)
    comp_motion = np.array(comp_motion)
    trajectory = np.cumsum(comp_motion, axis=0)

    kernal = gaussian_1d(100, 1000)
    smooth_trajectory = smooth(trajectory, kernal)
    smooth_motion = comp_motion + smooth_trajectory - trajectory
    # (c) Frame Correction
    prev_frame = reader.getFrame(0)
    # write
    path = "../Output/{}_{}/{}.avi".format("1", "B", vid)
    path_check = "../Output/{}_{}".format("1", "B")
    if os.path.exists(path_check) == False:
        os.mkdir(path_check)
    writer = VideoWriter(path, reader.getFPS())
    frame_write = cv2.hconcat([prev_frame, prev_frame])
    writer.writeFrame(frame_write)
    for i in range(1, reader.nrFrames - 1):
        current_frame = reader.getFrame(i)
        dx = smooth_motion[i, 0]
        dy = smooth_motion[i, 1]
        d_theta = smooth_motion[i, 2]
        # Reconstruct M matrix:
        M = np.zeros((2, 3), np.float32)
        M[1, 1] = M[0, 0] = np.cos(d_theta)
        M[0, 1] = -np.sin(d_theta)
        M[1, 0] = np.sin(d_theta)
        M[0, 2] = dx
        M[1, 2] = dy
        frame_stabilized = cv2.warpAffine(current_frame, M, (reader.width, reader.height))
        # Uncomment below code to remove the black pixels from boundary
        # frame_stabilized = cropify(frame_stabilized)
        frame_write = cv2.hconcat([current_frame, frame_stabilized])
        writer.writeFrame(frame_write)

    writer.close()
    end = time.time()
    print("Time taken per frame for 1B: {:0.2f} seconds".format((end - start) / reader.nrFrames))

def Q2_A_sep(reader, vid):
    # Intialize
    print("Saving the video with feature points marked in red and green...")
    prev_frame = reader.getFrame(0)
    n_total = 10000
    n_best = 1000
    path = "../Output/{}_{}/{}_sep.avi".format("2", "A", vid)
    path_check = "../Output/{}_{}".format("2", "A")
    if os.path.exists(path_check) == False:
        os.mkdir(path_check)
    writer = VideoWriter(path, reader.getFPS())

    # (a). Get Compact Motion Features
    for i in range(1, reader.nrFrames):
        current_frame = reader.getFrame(i)
        start = time.time()
        # Get keypoints and descriptor using ORB for previous and Next frame
        kp1, kp2 = fast_detector(prev_frame, current_frame, n_total)
        des1, des2 = brief_descriptor(prev_frame, kp1, current_frame, kp2)
        # Brute force matcher to find the correspondence
        brute_force_obj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        total_matches = brute_force_obj.match(des1, des2)
        best_matches = sorted(total_matches, key=lambda x: x.distance)[0:n_best]
        query_indices = [x.queryIdx for x in best_matches]
        target_indices = [x.trainIdx for x in best_matches]
        # Key-point coordinate corresponding to the indices
        kp1_best = [kp1[i] for i in query_indices]
        kp2_best = [kp2[i] for i in target_indices]
        # Generating the set to be used for finding the motion parameters
        points1 = [x.pt for x in kp1_best]
        points2 = [x.pt for x in kp2_best]
        arr_pt1 = np.array(points1)
        arr_pt2 = np.array(points2)
        arr_pt1 = arr_pt1.astype('int32')
        arr_pt2 = arr_pt2.astype('int32')

        t = 2.5
        _, mask = cv2.findHomography(arr_pt1, arr_pt2, cv2.RANSAC, t)

        image = current_frame.copy()
        for i in range(mask.shape[0]):
            if(mask[i]==1):
                cv2.circle(image, (arr_pt1[i][0], arr_pt1[i][1]), radius=3, color=(0, 255, 0), thickness=-1)
            else:
                cv2.circle(image, (arr_pt1[i][0], arr_pt1[i][1]), radius=3, color=(255, 0, 0), thickness=-1)

        # write
        frame_write = cv2.hconcat([current_frame, image])
        writer.writeFrame(frame_write)
        # writer.writeFrame(image)
        prev_frame = current_frame
    writer.close()
    end = time.time()
    print("Total time taken to save the segmented Video: {:0.2f} seconds".format(end - start))

def Q2_A(reader, vid):
    # Intialize
    start = time.time()
    prev_frame = reader.getFrame(0)
    n_total = 800
    n_best = 50
    comp_motion = []  # To store compact motion features for each frame

    # (a). Get Compact Motion Features
    for i in range(1, reader.nrFrames):
        current_frame = reader.getFrame(i)
        # Get keypoints and descriptor using ORB for previous and Next frame
        kp1, kp2 = fast_detector(prev_frame, current_frame, n_total)
        des1, des2 = brief_descriptor(prev_frame, kp1, current_frame, kp2)
        # Brute force matcher to find the correspondence
        brute_force_obj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        total_matches = brute_force_obj.match(des1, des2)
        best_matches = sorted(total_matches, key=lambda x: x.distance)[0:n_best]
        query_indices = [x.queryIdx for x in best_matches]
        target_indices = [x.trainIdx for x in best_matches]
        # Key-point coordinate corresponding to the indices
        kp1_best = [kp1[i] for i in query_indices]
        kp2_best = [kp2[i] for i in target_indices]
        # Generating the set to be used for finding the motion parameters
        points1 = [x.pt for x in kp1_best]
        points2 = [x.pt for x in kp2_best]
        arr_pt1 = np.array(points1)
        arr_pt2 = np.array(points2)
        arr_pt1 = arr_pt1.astype('int32')
        arr_pt2 = arr_pt2.astype('int32')

        t = 2.7
        _, mask = cv2.findHomography(arr_pt1, arr_pt2, cv2.RANSAC, t)

        kp1_ransac = arr_pt1[mask.astype('bool').reshape(-1)]
        kp2_ransac = arr_pt2[mask.astype('bool').reshape(-1)]

        M = cv2.estimateRigidTransform(kp1_ransac, kp2_ransac, fullAffine=False)
        # dx, dy
        dx = M[0, 2]
        dy = M[1, 2]
        # change in angle d_theta
        d_theta = np.arctan2(M[1, 0], M[0, 0])
        comp_motion.append([dx, dy, d_theta])

        prev_frame = current_frame

    # (b). Smoothing the parametes
    comp_motion = np.array(comp_motion)
    trajectory = np.cumsum(comp_motion, axis=0)
    kernal = gaussian_1d(10, 100)
    smooth_trajectory = smooth(trajectory, kernal)
    smooth_motion = comp_motion + smooth_trajectory - trajectory
    # (c) Frame Correction
    prev_frame = reader.getFrame(0)
    # write
    # Check for Path and create the path
    path = "../Output/{}_{}/{}.avi".format("2", "A", vid)
    path_check = "../Output/{}_{}".format("2", "A")
    if os.path.exists(path_check) == False:
        os.mkdir(path_check)
    # VideoWriter object
    writer = VideoWriter(path, reader.getFPS())
    # Write input and output Video Side by side for comparison
    frame_write = cv2.hconcat([prev_frame, prev_frame])
    writer.writeFrame(frame_write)
    for i in range(1, reader.nrFrames - 1):
        current_frame = reader.getFrame(i)
        dx = smooth_motion[i, 0]
        dy = smooth_motion[i, 1]
        d_theta = smooth_motion[i, 2]
        # Reconstruct M matrix:
        M = np.zeros((2, 3), np.float32)
        M[1, 1] = M[0, 0] = np.cos(d_theta)
        M[0, 1] = -np.sin(d_theta)
        M[1, 0] = np.sin(d_theta)
        M[0, 2] = dx
        M[1, 2] = dy
        frame_stabilized = cv2.warpAffine(current_frame, M, (reader.width, reader.height))
        # Uncomment Below Code for Cropping the black pixels at boundary
        # frame_stabilized = cropify(frame_stabilized)

        frame_write = cv2.hconcat([current_frame, frame_stabilized])
        writer.writeFrame(frame_write)
        # Show the Video
    writer.close()
    end = time.time()
    print("Time taken per frame for for 2A: {:0.2f} seconds".format((end - start) / reader.nrFrames))

def Q2_B_sep(reader, vid):
    start = time.time()
    prev_frame = reader.getFrame(0)

    path = "../Output/{}_{}/{}_sep.avi".format("2", "B", vid)
    path_check = "../Output/{}_{}".format("2", "B")
    if os.path.exists(path_check) == False:
        os.mkdir(path_check)
    writer = VideoWriter(path, reader.getFPS())
    comp_motion = []
    # (a). Get Compact Motion Features
    for i in range(1, reader.nrFrames):
        current_frame = reader.getFrame(i)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        image_loc_1 = np.zeros_like(flow)
        image_loc_2 = np.zeros_like(flow)
        image_loc_1[:, :, 0] = np.arange(w)
        image_loc_1[:, :, 1] = np.arange(h)[:, np.newaxis]
        image_loc_2[:, :, 0] = flow[:, :, 0] + np.arange(w)
        image_loc_2[:, :, 1] = flow[:, :, 1] + np.arange(h)[:, np.newaxis]
        image_loc_1 = image_loc_1.reshape(h * w, 2)
        image_loc_2 = image_loc_2.reshape(h * w, 2)

        t = 2.1
        _, mask = cv2.findHomography(image_loc_2, image_loc_1, cv2.RANSAC, t)
        # print(mask)
        image = prev_frame.copy()
        image_loc_1 = image_loc_1.astype('int32')
        image_loc_2 = image_loc_2.astype('int32')
        for i in range(mask.shape[0]):
            if(mask[i]==1):
                image[image_loc_1[i][1], image_loc_1[i][0]] = [255, 255, 255]
            else:
                image[image_loc_1[i][1], image_loc_1[i][0]] = [0, 0, 0]

        frame_write = cv2.hconcat([current_frame, image])
        writer.writeFrame(frame_write)
        prev_frame = current_frame
    writer.close()
    end = time.time()
    print("Total time elapsed in writing the masked frames:{:0.2f} seconds".format(end-start))

def Q2_B(reader, vid):
    start = time.time()
    prev_frame = reader.getFrame(0)
    path = "../Output/{}_{}/{}.avi".format("2", "B", vid)
    path_check = "../Output/{}_{}".format("2", "B")
    if os.path.exists(path_check) == False:
        os.mkdir(path_check)
    comp_motion = []
    # (a). Get Compact Motion Features
    for i in range(1, reader.nrFrames):
        current_frame = reader.getFrame(i)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        flow_1 = np.zeros_like(flow)  # ,dtype='int32')
        flow_2 = np.zeros_like(flow)  # ,dtype='int32')
        flow_1[:, :, 0] = np.arange(w)
        flow_1[:, :, 1] = np.arange(h)[:, np.newaxis]
        flow_2[:, :, 0] = flow[:, :, 0] + np.arange(w)
        flow_2[:, :, 1] = flow[:, :, 1] + np.arange(h)[:, np.newaxis]
        flow_1 = flow_1.reshape(h * w, 2)
        flow_2 = flow_2.reshape(h * w, 2)

        t = 2.1
        _, mask = cv2.findHomography(flow_2, flow_1, cv2.RANSAC, t)

        kp1_ransac = flow_1[mask.astype('bool').reshape(-1)]
        kp2_ransac = flow_2[mask.astype('bool').reshape(-1)]

        M = cv2.estimateRigidTransform(kp1_ransac, kp2_ransac, fullAffine=False)
        # dx, dy
        dx = M[0, 2]
        dy = M[1, 2]
        # change in angle d_theta
        d_theta = np.arctan2(M[1, 0], M[0, 0])
        comp_motion.append([dx, dy, d_theta])

        prev_frame = current_frame

    # (b). Smoothing the parametes
    comp_motion = np.array(comp_motion)
    trajectory = np.cumsum(comp_motion, axis=0)

    kernal = gaussian_1d(10, 1000)
    smooth_trajectory = smooth(trajectory, kernal)
    smooth_motion = comp_motion + smooth_trajectory - trajectory
    # (c) Frame Correction
    prev_frame = reader.getFrame(0)
    # write
    # Check for Path and create the path
    path = "../Output/{}_{}/{}.avi".format("2", "B", vid)
    path_check = "../Output/{}_{}".format("2", "B")
    if os.path.exists(path_check) == False:
        os.mkdir(path_check)
    # VideoWriter object
    writer = VideoWriter(path, reader.getFPS())
    # Write input and output Video Side by side for comparison
    frame_write = cv2.hconcat([prev_frame, prev_frame])
    writer.writeFrame(frame_write)
    for i in range(1, reader.nrFrames - 1):
        current_frame = reader.getFrame(i)
        dx = smooth_motion[i, 0]
        dy = smooth_motion[i, 1]
        d_theta = smooth_motion[i, 2]
        # Reconstruct M matrix:
        M = np.zeros((2, 3), np.float32)
        M[1, 1] = M[0, 0] = np.cos(d_theta)
        M[0, 1] = -np.sin(d_theta)
        M[1, 0] = np.sin(d_theta)
        M[0, 2] = dx
        M[1, 2] = dy
        frame_stabilized = cv2.warpAffine(current_frame, M, (reader.width, reader.height))
        # Uncomment Below Code for Cropping the black pixels at boundary
        # frame_stabilized = cropify(frame_stabilized)

        frame_write = cv2.hconcat([current_frame, frame_stabilized])
        writer.writeFrame(frame_write)
    cv2.destroyAllWindows()
    writer.close()
    end = time.time()
    print("Time Taken per frame for 2B:{:0.2f} seconds". format((end-start)/reader.nrFrames))
    return

def Q2_C_sep(reader, vid):
    start = time.time()
    prev_frame = reader.getFrame(0)

    path = "../Output/{}_{}/{}_sep.avi".format("2", "C", vid)
    path_check = "../Output/{}_{}".format("2", "C")
    if os.path.exists(path_check) == False:
        os.mkdir(path_check)
    writer = VideoWriter(path, reader.getFPS())
    comp_motion = []
    # (a). Get Compact Motion Features
    for i in range(1, reader.nrFrames-1):
        current_frame = reader.getFrame(i)
        flow = reader.getFlow(i)
        h, w = flow.shape[:2]
        flow_1 = np.zeros_like(flow)
        flow_2 = np.zeros_like(flow)
        flow_1[:, :, 0] = np.arange(w)
        flow_1[:, :, 1] = np.arange(h)[:, np.newaxis]
        flow_2[:, :, 0] = flow[:, :, 0] + np.arange(w)
        flow_2[:, :, 1] = flow[:, :, 1] + np.arange(h)[:, np.newaxis]
        flow_1 = flow_1.reshape(h * w, 2)
        flow_2 = flow_2.reshape(h * w, 2)

        t = 2.1
        _, mask = cv2.findHomography(flow_2, flow_1, cv2.RANSAC, t)
        image = prev_frame.copy()
        flow_1 = flow_1.astype('int32')
        flow_2 = flow_2.astype('int32')
        for i in range(mask.shape[0]):
            if(mask[i]==1):
                image[flow_1[i][1], flow_1[i][0]] = [255, 255, 255]
            else:
                image[flow_1[i][1], flow_1[i][0]] = [0, 0, 0]
        frame_write = cv2.hconcat([current_frame, image])
        writer.writeFrame(frame_write)
        prev_frame = current_frame
    writer.close()
    end = time.time()
    print("Time taken to save the saving the masks:{:0.2f} sec".format(end-start))

def Q2_C(reader, vid):
    start = time.time()
    path = "../Output/{}_{}/{}.avi".format("2", "C", vid)
    path_check = "../Output/{}_{}".format("2", "C")
    if os.path.exists(path_check) == False:
        os.mkdir(path_check)
    comp_motion = []
    # (a). Get Compact Motion Features
    for i in range(1, reader.nrFrames-1):
        flow = reader.getFlow(i)
        h, w = flow.shape[:2]
        image_loc_1 = np.zeros_like(flow)
        image_loc_2 = np.zeros_like(flow)
        image_loc_1[:, :, 0] = np.arange(w)
        image_loc_1[:, :, 1] = np.arange(h)[:, np.newaxis]
        image_loc_2[:, :, 0] = flow[:, :, 0] + np.arange(w)
        image_loc_2[:, :, 1] = flow[:, :, 1] + np.arange(h)[:, np.newaxis]
        image_loc_1 = image_loc_1.reshape(h * w, 2)
        image_loc_2 = image_loc_2.reshape(h * w, 2)

        t = 2.1
        _, mask = cv2.findHomography(image_loc_2, image_loc_1, cv2.RANSAC, t)

        kp1_ransac = image_loc_1[mask.astype('bool').reshape(-1)]
        kp2_ransac = image_loc_2[mask.astype('bool').reshape(-1)]

        M = cv2.estimateRigidTransform(kp1_ransac, kp2_ransac, fullAffine=False)
        # dx, dy
        dx = M[0, 2]
        dy = M[1, 2]
        # change in angle d_theta
        d_theta = np.arctan2(M[1, 0], M[0, 0])
        comp_motion.append([dx, dy, d_theta])

    # (b). Smoothing the parametes
    comp_motion = np.array(comp_motion)
    trajectory = np.cumsum(comp_motion, axis=0)

    kernal = gaussian_1d(10, 1000)
    smooth_trajectory = smooth(trajectory, kernal)
    smooth_motion = comp_motion + smooth_trajectory - trajectory
    # (c) Frame Correction
    prev_frame = reader.getFrame(0)
    # write
    # Check for Path and create the path
    path = "../Output/{}_{}/{}.avi".format("2", "C", vid)
    path_check = "../Output/{}_{}".format("2", "C")
    if os.path.exists(path_check) == False:
        os.mkdir(path_check)
    # VideoWriter object
    writer = VideoWriter(path, reader.getFPS())
    # Write input and output Video Side by side for comparison
    frame_write = cv2.hconcat([prev_frame, prev_frame])
    writer.writeFrame(frame_write)
    for i in range(1, reader.nrFrames - 2):
        current_frame = reader.getFrame(i)
        dx = smooth_motion[i, 0]
        dy = smooth_motion[i, 1]
        d_theta = smooth_motion[i, 2]
        # Reconstruct M matrix:
        M = np.zeros((2, 3), np.float32)
        M[1, 1] = M[0, 0] = np.cos(d_theta)
        M[0, 1] = -np.sin(d_theta)
        M[1, 0] = np.sin(d_theta)
        M[0, 2] = dx
        M[1, 2] = dy
        frame_stabilized = cv2.warpAffine(current_frame, M, (reader.width, reader.height))
        # Uncomment Below Code for Cropping the black pixels at boundary
        # frame_stabilized = cropify(frame_stabilized)

        frame_write = cv2.hconcat([current_frame, frame_stabilized])
        writer.writeFrame(frame_write)
    cv2.destroyAllWindows()
    writer.close()
    end = time.time()
    print("Time Taken per frame for 2C:{:0.2f} sec". format((end-start)/reader.nrFrames))
    return

if __name__=="__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-question", "--question", type = str, required = True, help="Question Number 1 or 2")
    parser.add_argument("-subQuestion", "--subQuestion", type = str, required = True, help="sub question Number A or B")
    parser.add_argument("-video", "--video", type = int, required = True, help="Path to video (data/Q#/?.avi)")
    args = vars(parser.parse_args())
    q = args["question"]
    sq = args["subQuestion"]
    vid = args["video"]

    # Sanity check for all the input arguments
    if q not in ["1","2", "3"]:
        print("Question can only be 1 2 or 3")
        exit()
    if sq not in ["A", "B", "C"]:
        print("Subquestion must only be A, B or C")
        exit()
    video_path = "..\data\Q" + str(q) + "\\" + str(vid) +".avi"
    if os.path.exists(video_path) == False:
        print("Video Path is wrong. Please retry with correct path")
        exit()
    # VideoUtils object for reader
    reader = VideoReader(DatasetPath="..\data", question = q, vidId=vid)
    # Function call based on question and subquestion
    call = q+"_"+sq
    function_call_dict = {
        "1_A": Q1_A,
        "1_B": Q1_B,
        "2_A": Q2_A,
        "2_B": Q2_B,
        "2_C": Q2_C,
        "2_A_sep": Q2_A_sep,
        "2_B_sep": Q2_B_sep,
        "2_C_sep": Q2_C_sep,
    }
    if os.path.exists("../Output") == False:
        os.mkdir("../Output")
    path = "../Output/{}_{}/{}.avi".format(q, sq, vid)
    function_call_dict[call](reader, vid)
    print("Output is saved in '{}' directory".format(path))
    if(q == '2'):
        call = q+"_"+sq+ "_sep"
        function_call_dict[call](reader, vid)
        path = "../Output/{}_{}/{}_sep.avi".format(q, sq, vid)
        print("Output for Segmented Keypoints/ Video is saved in '{}' directory".format(path))









