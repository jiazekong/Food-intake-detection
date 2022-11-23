# 递归获取.csv文件存入到list1
import os
import pandas as pd
import numpy as np
import csv


#  Put all the .csv files in the list_csv
def _list_dir(file_dir,processed_csv):
    
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        # judge if it is the file
        if os.path.isfile(path):
         
         # if it is .csv file, if it is put the dir into the list_csv
         if os.path.splitext(path)[1] == '.csv':
            if path.find("inertial_processed") != -1:
                id = cur_file[:6] #get the id of the people, for example 1001_1
                processed_csv[id]=path # a dictionary stores the id and path of the processed file
            

        elif os.path.isdir(path):
            _list_dir(path,processed_csv)
    return processed_csv

def _splitdataset(processed_csv):#split the files into test, vlidation, and train. return all the catigories
    
    division = _read_division()
    dataset_train, dataset_test, dataset_vali = [],[],[]
    for id, path in processed_csv.items():
        type = division[id]
        if type == 'TRAIN':
            dataset_train.append(path)
        elif type == 'TEST':
            dataset_test.append(path)
        elif type == 'VALID':
            dataset_vali.append(path)

    return dataset_train, dataset_test, dataset_vali

def _read_csvfile(processed_files): #add each colume data together and combine these colume together

    right_acc_x, right_acc_y, right_acc_z = [], [], []
    right_gyro_x, right_gyro_y, right_gyro_z = [], [], []
    label = []
    number = len(processed_files)
    for i in range(number):
        file = pd.read_csv(processed_files[i])

        right_acc_x = np.concatenate([right_acc_x, file["dom_acc_x"]])
        right_acc_y = np.concatenate([right_acc_y, file["dom_acc_y"]])
        right_acc_z = np.concatenate([right_acc_z, file["dom_acc_z"]])
        right_gyro_x = np.concatenate([right_gyro_x, file["dom_gyro_x"]])
        right_gyro_y = np.concatenate([right_gyro_y, file["dom_gyro_y"]])
        right_gyro_z = np.concatenate([right_gyro_z, file["dom_gyro_z"]])
        tempLabel = file["label_1"]
        tempLabel = np.array((tempLabel=="Intake").astype(int))
        label = np.concatenate([label,tempLabel])

        
    label = np.array(label).reshape(1,-1)
    right_acc = np.vstack((right_acc_x, right_acc_y,right_acc_z))
    right_gyro = np.vstack((right_gyro_x,right_gyro_y,right_gyro_z))
    data = np.vstack((right_acc,right_gyro))
    
    return right_acc, right_gyro, label, data

def _read_division():
    path = r'/Users/yutongcai/Desktop/public eat dataset/OREBA_Dataset_Public_1_0/Dataset_Public/oreba_dis/recordings.csv'
    recordings = pd.read_csv(path)
    id = list(recordings["Recording_ID"])
    split = list(recordings["Split"])
    division = dict(zip(id,split))
    return division




def get_dataset():
    paths = r'/Users/yutongcai/Desktop/public eat dataset/OREBA_Dataset_Public_1_0/Dataset_Public/oreba_dis/recordings' #E:\论文数据集\OREBA_Dataset_Public_1_0\Dataset_Public\oreba_dis\recordings
    processed_csv = {}
    _list_dir(file_dir=paths,processed_csv=processed_csv)
    dataset_train,dataset_test,dataset_vali = _splitdataset(processed_csv)
    test_acc,test_gyro,test_label,test_data = _read_csvfile(dataset_test)
    train_acc, train_gyro, train_label,train_data = _read_csvfile(dataset_train)
    vali_acc, vali_gyro, vali_label,vali_data = _read_csvfile(dataset_vali)
    return test_data,test_label, train_data,train_label, vali_data, vali_label