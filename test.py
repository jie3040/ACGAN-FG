from keras.models import load_model

import tensorflow as tf
import numpy as np
import random
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate,multiply
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv1D,MaxPooling1D,UpSampling1D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D,Embedding,concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LeakyReLU
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def scalar_stand(Train_X, Test_X):
    # 用训练集标准差标准化训练集以及测试集
    scalar_train = preprocessing.StandardScaler().fit(Train_X)
    #scalar_test = preprocessing.StandardScaler().fit(Test_X)
    Train_X = scalar_train.transform(Train_X)
    Test_X = scalar_train.transform(Test_X)
    return Train_X, Test_X


def feature_generation_and_diagnosis(add_quantity,test_x,test_y,autoencoder,generator,classifier):
    
    
    Labels_train=[]
    Labels_test=[]
    Generated_feature=[]

    for j in range(3):

      i=960*j

      attribute_vector=test_y[i]

      #attribute_list=[item for sublist in attribute_vector for item in sublist]

      attribute=[attribute_vector for _ in range(add_quantity)]
      attribute=np.array(attribute)
     
      print(attribute.shape)
           
      noise_shape=(add_quantity, 50, 1)
      noise = tf.random.normal(shape=noise_shape)

      generated_feature=generator.predict([noise,attribute])
      
      Generated_feature.append(generated_feature)

      labels_train = np.full((add_quantity, 1), j)
      labels_test = np.full((960, 1), j)

      Labels_train.append(labels_train)
      Labels_test.append(labels_test)
    
    Generated_feature=np.array(Generated_feature).reshape(-1, 256)
    Labels_train=np.array(Labels_train).reshape(-1, 1)

    Labels_test=np.array(Labels_test).reshape(-1, 1)  
    test_feature, decoded_test= autoencoder(test_x)

    hidden_ouput_train,predict_attribute_train=classifier(Generated_feature)
    new_feature_train=np.concatenate((Generated_feature, hidden_ouput_train), axis=1)

    hidden_ouput_test,predict_attribute_test=classifier(test_feature)
    new_feature_test=np.concatenate((test_feature, hidden_ouput_test), axis=1)


    
    train_X=new_feature_train
    train_Y=Labels_train
    
    test_X=new_feature_test
    test_Y=Labels_test

    train_X,test_X=scalar_stand(train_X, test_X)



    classifier_lsvm = LinearSVC()

    classifier_lsvm.fit(train_X, train_Y)
    
    Y_pred_lsvm = classifier_lsvm.predict(test_X)

    accuracy_lsvm = accuracy_score(test_Y, Y_pred_lsvm)
    
    
    
    classifier_nrf = RandomForestClassifier(n_estimators=100)
    
    classifier_nrf.fit(train_X, train_Y)
    
    Y_pred_nrf = classifier_nrf.predict(test_X)

    accuracy_nrf = accuracy_score(test_Y, Y_pred_nrf)
    
    
    classifier_pnb = GaussianNB()
    
    classifier_pnb.fit(train_X, train_Y)
    
    Y_pred_pnb = classifier_pnb.predict(test_X)

    accuracy_pnb = accuracy_score(test_Y, Y_pred_pnb)
    
    
    classifier_mlp = MLPClassifier(hidden_layer_sizes=(100, 50),
                               activation='relu',
                               solver='adam',
                               alpha=0.0001,
                               batch_size='auto',
                               learning_rate='constant',
                               max_iter=200,
                               tol=0.0001,
                               random_state=42)
    
    classifier_mlp.fit(train_X, train_Y)
    
    Y_pred_mlp = classifier_mlp.predict(test_X)

    accuracy_mlp = accuracy_score(test_Y, Y_pred_mlp)
   
    
    return accuracy_lsvm,accuracy_nrf,accuracy_pnb,accuracy_mlp
