# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:30:34 2020
Compute DDS between Taskonomy tasks
@author: kshitij
"""
import sys
sys.argv = ['']

# use if you use python interactive
# %reload_ext autoreload
# %autoreload 2

import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
import time
from sklearn.preprocessing import StandardScaler
from scipy.io import savemat
from pathlib import Path 

from utils import  get_similarity_from_rdms, get_similarity, rdm 

def get_features(features_filename):
    """

    Parameters
    ----------
    taskonomy_feats_path : TYPE
        DESCRIPTION.
    num_images : int
        number of images to compute DDS

    Returns
    -------
    taskonomy_data : dict
        dictionary containg features of taskonomy models.

    """
    if os.path.isfile(features_filename):
        start = time.time()
        taskonomy_data = np.load(features_filename,allow_pickle=True)
        end = time.time() 
        print("whole file loading time is ", end - start)
        taskonomy_data_full = taskonomy_data.item() 
    
    return taskonomy_data_full 



def split_dataset_train_test(task_list, data=None, train_area=(None,None), test_area=(None,None)):
    """[summary]

    Args:
        train_area ([type]): [description]
        dataset ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    train_images = {}
    test_images = {}
    beg_train, end_train  = train_area
    beg_test, end_test = test_area
    for task in tqdm(task_list):  # to separately train and test rdm´s 
        train_images[task] = data[task][beg_train:end_train,:]
        test_images[task] = data[task][beg_test:end_test,:]

    return train_images, test_images

def create_rdm(num_images, task_list, rdm_data, kernel_type, feature_norm_type, dist_type, save_dir, save_rdms): 
    """[summary]

    Args:
        task_list ([type]): [description]
        taskonomy_data_train ([type]): [description]
        taskonomy_data_test ([type]): [description]
        kernel_type ([type]): [description]
        feature_norm_type ([type]): [description]
        dist_type ([type]): [description]
        save_dir ([type]): [description]
    """
    
    save_path = os.path.join(save_dir)

    affinity_ablation = {}
    for dist in (dist_type):
        affinity_ablation[dist]={}
        for feature_norm in (feature_norm_type):
            rdm_matrix = np.zeros(len(task_list), dtype=object)

            method = dist + "__" + feature_norm
            start = time.time()

            for num in num_images: # because first key is number of images, that are used for train/test
                for index1,task1 in tqdm(enumerate(task_list)):  
                    x = StandardScaler().fit_transform(rdm_data[str(num)][task1])
                    rdm_matrix[index1] = rdm(x,dist)
           
                    # create directory for rdm if needed
                    Path(save_rdms+str(num)+"/").mkdir(parents=True, exist_ok=True) 

                    # save rdm to path defined  
                    np.save(save_rdms+str(num)+"/"+task1+"_yd_results", rdm_matrix[index1])

            end = time.time()
            print("Method is ", method)
            print("Time taken is ", end - start)
    np.save(save_path, affinity_ablation)


def main():
    parser = argparse.ArgumentParser(description='Computing Duality Diagram Similarity between Taskonomy Tasks')
    parser.add_argument('-d','--dataset', help='image dataset to use for computing DDS: options are [pascal_5000, taskonomy_5000, nyuv2]', default = "taskonomy_5000", type=str)
    parser.add_argument('-fd','--feature_dir', help='path to saved features from taskonomy models', default = "../../../data2/yd", type=str)
    parser.add_argument('-sd','--save_dir', help='path to save the DDS results', default = "./results/DDScomparison_taskonomy", type=str)
    parser.add_argument('-n','--num_images', help='number of images to compute DDS', default = 200, type=int)
    args = vars(parser.parse_args())

    # create rdms according to these tasks
    list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
    keypoint2d keypoint3d colorization \
    reshade rgb2depth rgb2mist rgb2sfnorm \
    room_layout segment25d segment2d vanishing_point \
    segmentsemantic class_1000 class_places inpainting_whole' 

    # choose parameters for further calculations 
    dataset = args["dataset"] 
    kernel_type = ['rbf','lap','linear'] # possible kernels (f in DDS)
    feature_norm_type = ['Znorm']  #['None','centering','znorm','group_norm','instance_norm','layer_norm','batch_norm'] # possible normalizations (Q,D in DDS)
    dist_type = ['cosine']  #, 'pearson', 'euclidean'] 
    feature_dir = args['feature_dir']
    save_dir = args['save_dir']
    task_list = list_of_tasks.split(' ')
    task_list = list(filter(None, task_list))

    # load and save directory 
    features_filename = os.path.join(feature_dir, "taskonomy_pascal_feats_" + dataset + ".npy")
    save_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 

    # get taskonomy data 
    taskonomy_data = get_features(features_filename)
    
    # store taskonomy rdm´s with different image-numbers  
    train_num = [50] #[200, 400, 500, 600, 800, 1000, 1200, 1400, 1500, 1600, 1800, 2000, 2200, 2400, 2500, 
               #3000, 3500, 4000, 4500] # amount used for train and test 
    test_num = [50] #[500, 200, 400, 500, 600, 800, 1000, 1200, 1400, 1500, 1600, 1800, 2000, 2200, 2400, 2500]
    rdm_train_per_img_size = {}  # nested dict, with number of img as key for one rdm 
    rdm_test_per_img_size = {} 
    train_save_path = "./yd_results/yd_train/"  # save train-rdms in this path
    test_save_path = "./yd_results/yd_test/" # save test-rdms in this path

    for test_img in tqdm(test_num):
        _, taskonomy_data_test = split_dataset_train_test(task_list, data=taskonomy_data, test_area=(test_img,2*test_img)) # function that returns features from taskonomy models for first #num_images
        rdm_test_per_img_size[str(test_img)] = taskonomy_data_test

    for train_img in tqdm(train_num): 
        taskonomy_data_train, _ = split_dataset_train_test(task_list, data=taskonomy_data, train_area=(0,train_img)) # function that returns features from taskonomy models for first #num_images
        rdm_train_per_img_size[str(train_img)] = taskonomy_data_train 
    
    # create and save train and test RDM´s with different number of images 
    create_rdm(train_num, task_list, rdm_train_per_img_size, kernel_type, feature_norm_type, dist_type, save_dir, save_rdms=train_save_path)
    create_rdm(test_num, task_list, rdm_test_per_img_size, kernel_type, feature_norm_type, dist_type, save_dir, save_rdms=test_save_path)


if __name__ == "__main__":
    main()