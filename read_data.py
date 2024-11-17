
import pandas as pd
import numpy as np

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
        testlabel += [item] * 480
        test_attributelabel += [attribute_matrix[item, :]] * 480
        testdata.append(data_list[item])
    testlabel = np.row_stack(testlabel)
    test_attributelabel = np.row_stack(test_attributelabel)
    testdata = np.row_stack(testdata)

    return traindata, trainlabel, train_attributelabel, \
           testdata, testlabel, test_attributelabel, \
           attribute_matrix_.iloc[test_index,:], attribute_matrix_.iloc[train_index, :]

traindata, trainlabel, train_attributelabel,\
testdata, testlabel, test_attributelabel, \
test_attribute_matrix, train_attribute_matrix = creat_dataset([1, 6, 14])



