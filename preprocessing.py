# 递归获取.csv文件存入到list1
import os
import pandas as pd
import numpy as np
import csv

TRAIN_IDS = [0,1,2,3,6,7]
VALID_IDS = [4,8]
TEST_IDS = [5,9]
#  Put all the .csv files in the list_csv
def list_dir(file_dir):

    # list_csv = []
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        # judge if it is the file
        if os.path.isfile(path):

            dir_files = os.path.join(file_dir, cur_file)
        # if it is .csv file, if it is put the dir into the list_csv
        if os.path.splitext(path)[1] == '.csv':
            csv_file = os.path.join(file_dir, cur_file)
            list_csv.append(csv_file)

        if os.path.isdir(path):
            list_dir(path)
    return list_csv

def splitdataset(List_csv):#split the files into test, vlidation, and train. return all the catigories
    processed_files = List_csv[1::3]

    dataset_train, dataset_test, dataset_vali = [],[],[]

    for train in range(len(TRAIN_IDS)):
        position = TRAIN_IDS[train]
        dataset_train.append(processed_files[position])

    for test in range(len(TEST_IDS)):
        position = TEST_IDS[test]
        dataset_test.append(processed_files[position])

    for vali in range(len(VALID_IDS)):
        position = VALID_IDS[vali]
        dataset_vali.append(processed_files[position])
    return dataset_train, dataset_test, dataset_vali

def read_csvfile(processed_files): #add each colume data together and combine these colume together

    right_acc_x, right_acc_y, right_acc_z = [], [], []
    right_gyro_x, right_gyro_y, right_gyro_z = [], [], []
    Label = []
    number = len(processed_files)
    for i in range(1):
        file = pd.read_csv(processed_files[i])

        right_acc_x = np.concatenate([right_acc_x, file["dom_acc_x"]])
        right_acc_y = np.concatenate([right_acc_y, file["dom_acc_y"]])
        right_acc_z = np.concatenate([right_acc_z, file["dom_acc_z"]])
        right_gyro_x = np.concatenate([right_gyro_x, file["dom_gyro_x"]])
        right_gyro_y = np.concatenate([right_gyro_y, file["dom_gyro_y"]])
        right_gyro_z = np.concatenate([right_gyro_z, file["dom_gyro_z"]])

        with open(processed_files[i], 'rb') as f:
            num_rows = len(f.readlines())
            print(num_rows)
            for j in range(num_rows-1):
                if (str(file.iloc[j,16]) == "Idle"):
                    Label.append(0)
                elif(str(file.iloc[j,16]) == "Intake"):
                    Label.append(1)
                else:
                    pass
            print("the length of Lable of file is ", len(Label))

    right_acc = np.vstack((right_acc_x, right_acc_y,right_acc_z))
    right_gyro = np.vstack((right_gyro_x,right_gyro_y,right_gyro_z))
    data = np.vstack((right_acc,right_gyro))
    return right_acc, right_gyro, Label,data


if __name__ == '__main__':
    paths = r'E:\论文数据集\OREBA_Dataset_Public_1_0\Dataset_Public\oreba_dis\recordings'
    list_csv = []
    list_csv = list_dir(file_dir=paths)
    print(list_csv[0])
    file = pd.read_csv(list_csv[1])
    print(str(file.iloc[3]["dom_hand"]))
    dataset_train,dataset_test,dataset_vali = splitdataset(list_csv)
    print(len(dataset_vali))
    test_acc,test_gyro,test_label,data = read_csvfile(dataset_test)
    train_acc, train_gyro, train_label,data = read_csvfile(dataset_train)
    vali_acc, vali_gyro, vali_label,data = read_csvfile(dataset_vali)
    print(data.shape)



