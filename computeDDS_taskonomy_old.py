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

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'


def get_features(features_filename,num_images):
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
    task_list = list_of_tasks.split(' ')
    print("hi")
    print(features_filename)
    if os.path.isfile(features_filename):
        start = time.time()
        taskonomy_data = np.load(features_filename,allow_pickle=True)
        print("12342", taskonomy_data)
        end = time.time()
        print("whole file loading time is ", end - start)
        taskonomy_data_full = taskonomy_data.item()
        taskonomy_data_few_images_train = {}
        taskonomy_data_few_images_test = {}
        for index,task in enumerate(task_list):  # to separately train and test rdm´s 
            taskonomy_data_few_images_train[task] = taskonomy_data_full[task][:num_images,:]
            taskonomy_data_few_images_test[task] = taskonomy_data_full[task][4500:,:]         
        return taskonomy_data_few_images_train, taskonomy_data_few_images_test

def take_taskonomy_data(task_list, dataset, num_images, feature_dir, save_dir): 
    
    features_filename = os.path.join(feature_dir, "taskonomy_pascal_feats_" + dataset + ".npy")
    save_dir = os.path.join(save_dir,dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    
    # store taskonomy rdm´s with number of img as key for nested dict 
    num_img_per_train = {}  
    num_img_per_test = {} 
    for number in num_images: 
        taskonomy_data_train, taskonomy_data_test = get_features(features_filename,number) # function that returns features from taskonomy models for first #num_images
        num_img_per_train[str(number)] = taskonomy_data_train 
        num_img_per_test[str(number)] = taskonomy_data_test 

    return num_img_per_train, num_img_per_test

def create_rdm_train_test(task_list, taskonomy_data_train, taskonomy_data_test, kernel_type, feature_norm_type, dist_type, save_dir): 
    save_path = os.path.join(save_dir)

    affinity_ablation = {}
    for dist in (dist_type):
        affinity_ablation[dist]={}
        for feature_norm in (feature_norm_type):
            rdm_matrix_train = np.zeros(len(task_list), dtype=object)
            rdm_matrix_test = np.zeros(len(task_list), dtype=object)

            method = dist + "__" + feature_norm
            start = time.time()

            for num in taskonomy_data_train: # because first key is number of images, that are used for train/test
                for index1,task1 in tqdm(enumerate(task_list)): 
                    x_train = StandardScaler().fit_transform(taskonomy_data_train[num][task1])
                    x_test = StandardScaler().fit_transform(taskonomy_data_test[num][task1])
                    rdm_matrix_train[index1] = rdm(x_train,dist)
                    rdm_matrix_test[index1] = rdm(x_test,dist)

                    # create directory for rdm
                    Path("./results_yd/yd_train/"+num+"/").mkdir(parents=True, exist_ok=True) 
                    Path("./results_yd/yd_test/"+num+"/").mkdir(parents=True, exist_ok=True) 

                    # save rdm to path defined above 
                    np.save("./results_yd/yd_train/"+num+"/"+task1+"_yd_results", rdm_matrix_train[index1])
                    np.save("./results_yd/yd_test/"+num+"/"+task1+"_yd_results", rdm_matrix_test[index1])

            end = time.time()
            print("Method is ", method)
            print("Time taken is ", end - start)
    np.save(save_path, affinity_ablation)


def main():
    parser = argparse.ArgumentParser(description='Computing Duality Diagram Similarity between Taskonomy Tasks')
    parser.add_argument('-d','--dataset', help='image dataset to use for computing DDS: options are [pascal_5000, taskonomy_5000, nyuv2]', default = "taskonomy_5000", type=str)
    parser.add_argument('-fd','--feature_dir', help='path to saved features from taskonomy models', default = "features", type=str)
    parser.add_argument('-sd','--save_dir', help='path to save the DDS results', default = "./results/DDScomparison_taskonomy", type=str)
    parser.add_argument('-n','--num_images', help='number of images to compute DDS', default = 200, type=int)
    args = vars(parser.parse_args())

    dataset = args["dataset"] 
    kernel_type = ['rbf','lap','linear'] # possible kernels (f in DDS)
    feature_norm_type = ['Znorm']  #['None','centering','znorm','group_norm','instance_norm','layer_norm','batch_norm'] # possible normalizations (Q,D in DDS)
    dist_type = ['cosine']  #, 'pearson', 'euclidean']
    task_list = list_of_tasks.split(' ') 
    #number_img = [50, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400] # amount used for train and test 
    number_img = [500, 1500, 2500, 3000, 3500, 4000, 4500] # amount used for train and test 

    print(args['feature_dir'])
    taskonomy_data_trains, taskonomy_data_tests = take_taskonomy_data(task_list=task_list, dataset=dataset, num_images=number_img, feature_dir=args['feature_dir'], save_dir=args['save_dir'])  # take out amount of img of taskonomy 5000 dataset
    
    # Create train and test RDM´s with different number of images 
    create_rdm_train_test(task_list = task_list, taskonomy_data_train = taskonomy_data_trains, taskonomy_data_test = taskonomy_data_tests, kernel_type=kernel_type, feature_norm_type=feature_norm_type, dist_type=dist_type, save_dir=args['save_dir'])

if __name__ == "__main__":
    main()