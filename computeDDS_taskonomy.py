# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:30:34 2020
Compute DDS between Taskonomy tasks
@author: kshitij
"""

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
import time
from sklearn.preprocessing import StandardScaler
from scipy.io import savemat
from pathlib import Path 

from utils import  get_similarity_from_rdms,get_similarity, rdm 



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
    if os.path.isfile(features_filename):
        start = time.time()
        taskonomy_data = np.load(features_filename,allow_pickle=True)
        end = time.time()
        print("whole file loading time is ", end - start)
        taskonomy_data_full = taskonomy_data.item()
        taskonomy_data_few_images_train = {}
        taskonomy_data_few_images_test = {}
        for index,task in enumerate(task_list):  # to separately train and test rdm´s 
            taskonomy_data_few_images_train[task] = taskonomy_data_full[task][:num_images,:]
            taskonomy_data_few_images_test[task] = taskonomy_data_full[task][num_images:2*num_images,:]         
        return taskonomy_data_few_images_train, taskonomy_data_few_images_test

def main():
    parser = argparse.ArgumentParser(description='Computing Duality Diagram Similarity between Taskonomy Tasks')
    parser.add_argument('-d','--dataset', help='image dataset to use for computing DDS: options are [pascal_5000, taskonomy_5000, nyuv2]', default = "taskonomy_5000", type=str)
    parser.add_argument('-fd','--feature_dir', help='path to saved features from taskonomy models', default = "../../../data2/yd", type=str)
    parser.add_argument('-sd','--save_dir', help='path to save the DDS results', default = "./results/DDScomparison_taskonomy", type=str)
    parser.add_argument('-n','--num_images', help='number of images to compute DDS', default = 4500, type=int)
    args = vars(parser.parse_args())

    num_images = args['num_images']
    dataset = args['dataset']
    features_filename = os.path.join(args['feature_dir'],"taskonomy_pascal_feats_" + args['dataset'] + ".npy")
    save_dir = os.path.join(args['save_dir'],dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    task_list = list_of_tasks.split(' ')

    taskonomy_data, taskonomy_data_test = get_features(features_filename,num_images) # function that returns features from taskonomy models for first #num_images


    # setting up DDS using Q,D,f,g for kernels
    kernel_type = ['rbf','lap','linear'] # possible kernels (f in DDS)
    feature_norm_type = ['znorm'] # ['None','centering','znorm','group_norm','instance_norm','layer_norm','batch_norm'] # possible normalizations (Q,D in DDS)


    # save_path = os.path.join(save_dir,'kernels.npy')
    # affinity_ablation = {}
    # for kernel in (kernel_type):
    #     affinity_ablation[kernel]={}
    #     for feature_norm in feature_norm_type:
    #         affinity_matrix = np.zeros((len(task_list), len(task_list)), float)
    #         method = kernel + "__" + feature_norm
    #         start = time.time()
    #         for index1,task1 in tqdm(enumerate(task_list)):
    #             for index2,task2 in (enumerate(task_list)):
    #                 if index1 > index2:
    #                     continue
    #                 affinity_matrix[index1,index2] = get_similarity(taskonomy_data[task1],\
            #                                                         taskonomy_data[task2],\
            #                                                         kernel,feature_norm)
            #         affinity_matrix[index2,index1] = affinity_matrix[index1,index2]
            # end = time.time()
            # print("Method is ", method)
            # print("Time taken is ", end - start)
            # affinity_ablation[kernel][feature_norm] = affinity_matrix

       # np.save(save_path,affinity_ablation)

    # setting up DDS using Q,D,f,g for distance functions
    save_path = os.path.join(save_dir)
    dist_type = ['cosine']  #['pearson', 'euclidean', 'cosine']
    affinity_ablation = {}
    for dist in (dist_type):
        affinity_ablation[dist]={}
        for feature_norm in (feature_norm_type):
            #affinity_matrix = np.zeros((len(task_list), len(task_list)), dtype = np.str)
            rdm_matrix_train = np.zeros(len(task_list), dtype=object)
            rdm_matrix_test = np.zeros(len(task_list), dtype=object)

            method = dist + "__" + feature_norm
            start = time.time()

            for index1,task1 in tqdm(enumerate(task_list)):
                # for index2,task2 in enumerate(task_list):
                #     if index1 > index2:
                #         continue
                #    affinity_matrix[index1,index2] = get_similarity_from_rdms(taskonomy_data[task1],\
                #                                                               taskonomy_data[task2],\
                #                                                               dist,feature_norm)
                #print("1000: ", taskonomy_data[task1], dist)
                #affinity_matrix[index1, index1] = 
                x_train = StandardScaler().fit_transform(taskonomy_data[task1])
                x_test = StandardScaler().fit_transform(taskonomy_data_test[task1])
                #print(type(rdm(taskonomy_data[task1],dist)))
                rdm_matrix_train[index1] = rdm(x_train,dist)
                rdm_matrix_test[index1] = rdm(x_test,dist)


                Path("./results_yd/yd_train/"+str(num_images)+"/").mkdir(parents=True, exist_ok=True) 
                Path("./results_yd/yd_test/"+str(num_images)+"/").mkdir(parents=True, exist_ok=True) 

                np.save("./results_yd/yd_train/"+str(num_images)+"/"+task1+"_yd_results", rdm_matrix_train[index1])
                np.save("./results_yd/yd_test/"+str(num_images)+"/"+task1+"_yd_results", rdm_matrix_test[index1])

                # create matlab format
                # savemat(, mdic)



            print("tasklist: ", task_list)
            print("len, tasklist: ", len(task_list))
            print("RDM: ", rdm_matrix_train)
            print("RDM_type: ", type(rdm_matrix_train))
            print("shape: ", rdm_matrix_train.shape)


            end = time.time()
            print("Method is ", method)
            print("Time taken is ", end - start)
    np.save(save_path, affinity_ablation)
    #np.save("./results_yd/task_rdms_train", rdm_matrix_train)
    #np.save("./results_yd/task_rdms_test", rdm_matrix_test)

if __name__ == "__main__":
    main()
