import numpy as np
from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm 

# # load rdm´s 
def compare_rdms(rdm_a, rdm_b):
    
    equal = {}
    for num_a, value_a in tqdm(rdm_a.items()):
        for num_b, value_b in rdm_b.items(): 
            if num_a == num_b:
                equal[num_b] = [value_a[i] == value_b[i] for i in range(len(value_a))]
                
    return equal


def main(): 

    # get directory of each rdm 
    yd_path_a = "yd_results/"
    yd_path_b = "../../../data2/yd/results_yd/" 
    dir_trains_a = [join(yd_path_a+"yd_train/", dir) for dir in listdir(yd_path_a+"yd_train") if isdir(join(yd_path_a+"yd_train/", dir))] 
    dir_tests_a = [join(yd_path_a+"yd_test/", dir) for dir in listdir(yd_path_a+"yd_test") if isdir(join(yd_path_a+"yd_test/", dir))] 
    dir_trains_b = [join(yd_path_b+"yd_train/", dir) for dir in listdir(yd_path_b+"yd_train") if isdir(join(yd_path_b+"yd_train/", dir))] 
    dir_tests_b = [join(yd_path_b+"yd_test/", dir) for dir in listdir(yd_path_b+"yd_test") if isdir(join(yd_path_b+"yd_test/", dir))] 

    # get rdms with various sizes
    num_trains_a = [num for num in listdir(yd_path_a+"yd_train") if isdir(join(yd_path_a+"yd_train/", num))] 
    num_tests_a = [num for num in listdir(yd_path_a+"yd_test") if isdir(join(yd_path_a+"yd_test/", num))] 
    num_trains_b = [num for num in listdir(yd_path_b+"yd_train") if isdir(join(yd_path_b+"yd_train/", num))] 
    num_tests_b = [num for num in listdir(yd_path_b+"yd_test") if isdir(join(yd_path_b+"yd_test/", num))] 

    # get npy files from each rdm for each rdm size
    train_tasks_a = {}
    for i, num in tqdm(enumerate(num_trains_a)):
        train_tasks_a[num] = [np.load(join(dir_trains_a[i],f)) for f in sorted(listdir(dir_trains_a[i]))]

    test_tasks_a = {}
    for i, num in tqdm(enumerate(num_tests_a)):
        test_tasks_a[num] = [np.load(join(dir_tests_a[i],f)) for f in sorted(listdir(dir_tests_a[i]))]

    train_tasks_b = {}
    for i, num in tqdm(enumerate(num_trains_b)):
        train_tasks_b[num] = [np.load(join(dir_trains_b[i],f)) for f in sorted(listdir(dir_trains_b[i]))]

    test_tasks_b = {}
    for i, num in tqdm(enumerate(num_tests_b)):
        test_tasks_b[num] = [np.load(join(dir_tests_b[i],f)) for f in sorted(listdir(dir_tests_b[i]))]


    wrong_train_rdms = compare_rdms(train_tasks_a, train_tasks_b) 
    wrong_test_rdms = compare_rdms(test_tasks_a, test_tasks_b) 


if __name__ == "__main__":
    main() 