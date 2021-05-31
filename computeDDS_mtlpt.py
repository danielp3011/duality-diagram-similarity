# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:30:34 2020
Compute DDS between Taskonomy tasks
@author: kshitij
"""


import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
import time
from sklearn.preprocessing import StandardScaler
from scipy.io import savemat
from pathlib import Path 
import sys
from tqdm import tqdm

from utils import  get_similarity_from_rdms,get_similarity, rdm 


def all_source_tasks_dic_generator(num_images, dir, out_name, tasks):
    file = os.path.join(dir, out_name+"_"+str(num_images)+"imgs.npy")
    print(file) 
    
    if os.path.exists(file):
        print("File already exists. File:", file,\
             "\nStopping creation.")
        return

    all_task_dictionary = {}

    for task in tasks:
        all_task_dictionary[task] = []        
    
    for task in tasks:
        print("Processing task: ", task)
        all_feats = os.listdir(os.path.join(dir, task))
        all_feats = all_feats[:num_images]
        for i, feat in tqdm(enumerate(all_feats)):
            all_task_dictionary[task].append(np.load(os.path.join(dir, task, feat), allow_pickle=True))

    for key, value in all_task_dictionary.items():
            all_task_dictionary[key] = np.array(value)
            print(all_task_dictionary[key].shape)
    
    print("44:", type(all_task_dictionary))

    print("Saving results...")
    np.save(os.path.join(dir, out_name+"_"+str(num_images)+"imgs.npy"), all_task_dictionary)
    
    return


def check_sizes(args):
    """Check if size for creating all task dictionary is enough for the split of train-and test-rdm´s

    Args:
        args ([argparse object]): [Variables coming from shell]
    """

    if args['num_images'] > args['num_imgs_features']/2:
        print("\nWarning!!! num_images is specified bigger than available data.\n")
        print("Terminating...")
        sys.exit()


def get_features(features_filename, num_images, task_list):
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
    print(features_filename)
    print("before if:")
    if os.path.isfile(features_filename):
        print("here")
        start = time.time()
        taskonomy_data = np.load(features_filename, allow_pickle=True)
        end = time.time()
        print("whole file loading time is ", end - start)
        print("type", type(taskonomy_data))
       
        taskonomy_data_full = taskonomy_data.item()
       
        taskonomy_data_few_images_train = {}
        taskonomy_data_few_images_test = {}
        for index,task in enumerate(task_list):  # to separately train and test rdm´s 
            taskonomy_data_few_images_train[task] = taskonomy_data_full[task][:num_images,:]
            taskonomy_data_few_images_test[task] = taskonomy_data_full[task][num_images:2*num_images,:]      

        print()   
        return taskonomy_data_few_images_train, taskonomy_data_few_images_test

def main():
    parser = argparse.ArgumentParser(description='Computing Duality Diagram Similarity between Pascal/NYUD Tasks')
    parser.add_argument('-d','--dataset', help='downsampled dataset to use for computing DDS: options are ["all_source_tasks_24-32", "all_source_tasks_48-64"]', default = "all_source_tasks_24-32", type=str)
    parser.add_argument('-fd','--feature_dir', help='Specify the path of the mode of the model, e.g. "/home/data2/yd/results_yd/mtlpt/PASCALContext/hrnet_w18/mti_net/aux_task=source_tasks/mti_net=Dsn_Dsl_Dhp=Dss" ', type=str)
    parser.add_argument('-sd','--save_dir', help='path to save the DDS results', type=str)
    parser.add_argument('-n','--num_images', help='number of images to compute DDS', default = 200, type=int)
    parser.add_argument('-nf', '--num_imgs_features' , help='number of images to add to the feature all_source dictionary', default=2000, type=int)
    parser.add_argument('-l', '--layer', help='Layer where the features are coming from, select from ["final", "scales"]', type=str)
    args = vars(parser.parse_args())

    check_sizes(args)  

    # selecting the layer of feature_maps
    if args['layer'] == "final":
        layer_list = [args['layer']]
    elif args['layer'] == "scales":
        # extracting the names of the folders for creating the layer_list: 
        layer_list = [f.path.rsplit("/",1)[1] for f in os.scandir(os.path.join(args['feature_dir'], "feature_maps", args['dataset'][-5:], args['layer'])) if f.is_dir()]
    # elif args['layer'] == "single_final": //TODO: Remove if really not necessary 
    #     layer_list = [args['layer']] 

    for layer in layer_list:
        print('\nProcessing layer ', layer, '.')
        if args['layer'] == "final":
            save_dir = os.path.join(args['save_dir'], "rdms", args['dataset'][-5:], args['layer'], str(args['num_images'])) 
            feature_dir = os.path.join(args['feature_dir'], "feature_maps", args['dataset'][-5:], args['layer'])
        elif args['layer'] == "scales":
            save_dir = os.path.join(args['save_dir'], "rdms", args['dataset'][-5:], args['layer'], layer, str(args['num_images'])) 
            feature_dir = os.path.join(args['feature_dir'], "feature_maps", args['dataset'][-5:], args['layer'], layer)
        # elif args['layer'] == "single_final": 
        #     save_dir = os.path.join(args['save_dir'], "rdms", args['dataset'][-5:], args['layer'], str(args['num_images']))
        #     feature_dir = 

        if not os.path.exists(save_dir):
            print("Creating save_dir")
            os.makedirs(save_dir)
        

        # get used tasks dynamically from path to features
        task_list = [f.path.rsplit("/",1)[1] for f in os.scandir(feature_dir) if f.is_dir()]

        # prepare all_source_tasks_[size].npy dictionary 
        all_source_tasks_dic_generator(args['num_imgs_features'], feature_dir, "all_source_tasks_" + str(args['dataset'][-5:]), task_list)
        
        
        print("Start loading the features for later rdm creation:")
        features_filename_to_load = os.path.join(feature_dir, args['dataset'] + '_' + str(args['num_imgs_features']) + 'imgs' + ".npy")  # deleted => "taskonomy_pascal_feats_" +
        print(features_filename_to_load, args['num_images'], task_list)
        # sys.exit()
        taskonomy_data, taskonomy_data_test = get_features(features_filename_to_load, args['num_images'], task_list) # function that returns features from taskonomy models for first #args['num_images']
        

        # setting up DDS using Q,D,f,g for kernels
        # kernel_type = ['rbf','lap','linear'] # possible kernels (f in DDS)
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
        for dist in tqdm((dist_type)):
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
                    print("here:")
                    print(taskonomy_data[task1])
                    print(taskonomy_data[task1].shape)

                    x_train = StandardScaler().fit_transform(taskonomy_data[task1])
                    x_test = StandardScaler().fit_transform(taskonomy_data_test[task1])
                    rdm_matrix_train[index1] = rdm(x_train,dist)
                    rdm_matrix_test[index1] = rdm(x_test,dist)

                    #Path(save_dir).mkdir(parents=True, exist_ok=True) 
                    #Path("./results_yd/yd_test/"+str(args['num_images'])+"/").mkdir(parents=True, exist_ok=True) 

                    # saving rdms:
                    np.save(save_dir + "/" + task1 + "_train", rdm_matrix_train[index1])
                    np.save(save_dir + "/" + task1 + "_test", rdm_matrix_test[index1])

                print("tasklist: ", task_list)
                print("len, tasklist: ", len(task_list))
                print("RDM: ", rdm_matrix_train)
                print("RDM_type: ", type(rdm_matrix_train))
                print("shape: ", rdm_matrix_train.shape)


                end = time.time()
                print("Method is ", method)
                print("Time taken is ", end - start)
        # np.save(save_path, affinity_ablation)


if __name__ == "__main__":
    main()
