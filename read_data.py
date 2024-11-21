
import pandas as pd
import numpy as np
from sklearn import preprocessing

def scalar_stand(Train_X, Test_X):
    scalar_train = preprocessing.StandardScaler().fit(Train_X)
    Train_X = scalar_train.transform(Train_X)
    Test_X = scalar_train.transform(Test_X)
    return Train_X, Test_X
   
def read_dat_file(file_path):
   
    try:
        data = np.loadtxt(file_path)
        return data
    except ValueError as e:
        print(f"Error loading file with np.loadtxt: {e}")
        print("Falling back to manual parsing...")
        
        # Fallback to manual parsing if the file format is irregular
        samples = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    # Split by whitespace and convert to float
                    elements = line.strip().split()
                    sample = [float(element) for element in elements]
                    samples.append(sample)
                except ValueError as e:
                    print(f"Skipping invalid line: {line.strip()}")
                    continue
                    
        return np.array(samples)

# data = read_dat_file('/home/liaowenjie/myfolder/GAN_for_UFD_3/re_imple/data/d01.dat')

# print(data)

def creat_dataset(test_index = [1, 6, 14]):
    path = '/home/liaowenjie/myfolder/GAN_for_UFD_3/re_imple/data/'
    print("loading data...")
    
    fault1 = read_dat_file(path + 'd01.dat')
    fault2 = read_dat_file(path + 'd02.dat')
    fault3 = read_dat_file(path + 'd03.dat')
    fault4 = read_dat_file(path + 'd04.dat')
    fault5 = read_dat_file(path + 'd05.dat')
    fault6 = read_dat_file(path + 'd06.dat')
    fault7 = read_dat_file(path + 'd07.dat')
    fault8 = read_dat_file(path + 'd08.dat')
    fault9 = read_dat_file(path + 'd09.dat')
    fault10 = read_dat_file(path + 'd10.dat')
    fault11 = read_dat_file(path + 'd11.dat')
    fault12 = read_dat_file(path + 'd12.dat')
    fault13 = read_dat_file(path + 'd13.dat')
    fault14 = read_dat_file(path + 'd14.dat')
    fault15 = read_dat_file(path + 'd15.dat')

    fault1_te = read_dat_file(path + 'd01_te.dat')
    fault2_te = read_dat_file(path + 'd02_te.dat')
    fault3_te = read_dat_file(path + 'd03_te.dat')
    fault4_te = read_dat_file(path + 'd04_te.dat')
    fault5_te = read_dat_file(path + 'd05_te.dat')
    fault6_te = read_dat_file(path + 'd06_te.dat')
    fault7_te = read_dat_file(path + 'd07_te.dat')
    fault8_te = read_dat_file(path + 'd08_te.dat')
    fault9_te = read_dat_file(path + 'd09_te.dat')
    fault10_te = read_dat_file(path + 'd10_te.dat')
    fault11_te = read_dat_file(path + 'd11_te.dat')
    fault12_te = read_dat_file(path + 'd12_te.dat')
    fault13_te = read_dat_file(path + 'd13_te.dat')
    fault14_te = read_dat_file(path + 'd14_te.dat')
    fault15_te = read_dat_file(path + 'd15_te.dat')
   
    attribute_matrix_ = pd.read_excel('/home/liaowenjie/myfolder/GAN_for_UFD_3/re_imple/attribute_matrix.xlsx', index_col='no')
    attribute_matrix = attribute_matrix_.values
    
    train_index = list(set(np.arange(15)) - set(test_index))
    
    test_index.sort()
    train_index.sort()
    
    print("test classes: {}".format(test_index))
    print("train classes: {}".format(train_index))
    
    data_list = [fault1, fault2, fault3, fault4, fault5,
                 fault6, fault7, fault8, fault9, fault10,
                 fault11, fault12, fault13, fault14, fault15]
    data_te_list = [fault1_te, fault2_te, fault3_te, fault4_te, fault5_te,
                 fault6_te, fault7_te, fault8_te, fault9_te, fault10_te,
                 fault11_te, fault12_te, fault13_te, fault14_te, fault15_te]
   
    trainlabel = []
    train_attributelabel = []
    traindata = []
    for item in train_index:
        trainlabel += [item] * 480
        train_attributelabel += [attribute_matrix[item, :]] * 480
        traindata.append(data_list[item])
    trainlabel = np.row_stack(trainlabel)
    train_attributelabel = np.row_stack(train_attributelabel)
    traindata = np.row_stack(traindata)

    testlabel = []
    test_attributelabel = []
    testdata = []
    for item in test_index:
        testlabel += [item] * 960
        test_attributelabel += [attribute_matrix[item, :]] * 960
        testdata.append(data_te_list[item])
    testlabel = np.row_stack(testlabel)
    test_attributelabel = np.row_stack(test_attributelabel)
    testdata = np.row_stack(testdata)

    return traindata, trainlabel, train_attributelabel, \
           testdata, testlabel, test_attributelabel, \
           attribute_matrix_.iloc[test_index,:], attribute_matrix_.iloc[train_index, :]

traindata, trainlabel, train_attributelabel,\
testdata, testlabel, test_attributelabel, \
test_attribute_matrix, train_attribute_matrix = creat_dataset([1, 6, 14])



