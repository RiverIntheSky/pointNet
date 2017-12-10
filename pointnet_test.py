#This is a Keras implementation of PointNet by Qi et al.
#https://github.com/charlesq34/pointnet/

import numpy as np
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import fileLoader as provider

point_num = 2048
num_labels = 3
segmentation = True
part_num=14

def generate_data(batch_size=32, n_samples=1024):
    out = np.zeros((batch_size, n_samples, 3))
    labels = np.zeros((batch_size, 1))   
        
    #########################################
    #obj dump
    #point_file = open("output/point_dump.obj","w")
    #mtl_file = open("output/point_dump.mtl","w")
    #point_file.write("mtllib point_dump.mtl\n")
    #########################################

    for i in range(batch_size):
        label = np.random.choice(num_labels)
        if label == 1: #sphere
            center = [0.5,0.5,0.5]   
            r = np.random.random(1) * 0.5         
            vecs = np.random.random((n_samples, 3)) * 2 - 1
            for j in range(n_samples):
                out[i,j, :] = np.transpose(center + r * vecs[j] / np.linalg.norm(vecs[j]))
            labels[i] = label
        elif label == 2:#cube
            center = [0.5,0.5,0.5]
            s = np.random.random(1) * 0.5
            origin = center - 0.5 * s
            vecs = np.random.random((n_samples, 3))
            for j in range(n_samples):
                if vecs[j,0] < 1.0 / 6.0 :
                    vecs[j,0] = 0
                    vecs[j,1] *= s
                    vecs[j,2] *= s
                    vecs[j] += origin 
                elif vecs[j,0] < 2.0 / 6.0 :
                    vecs[j,0] = s
                    vecs[j,1] *= s
                    vecs[j,2] *= s
                    vecs[j] += origin 
                elif vecs[j,0] < 3.0 / 6.0 :
                    vecs[j,0] = vecs[j,1] * s
                    vecs[j,1] = 0
                    vecs[j,2] *= s
                    vecs[j] += origin 
                elif vecs[j,0] < 4.0 / 6.0 :
                    vecs[j,0] = vecs[j,1] * s
                    vecs[j,1] = s
                    vecs[j,2] *= s
                    vecs[j] += origin 
                elif vecs[j,0] < 5.0 / 6.0 :
                    vecs[j,0] = vecs[j,2] * s
                    vecs[j,1] *= s
                    vecs[j,2] = 0 
                    vecs[j] += origin 
                else:
                    vecs[j,0] = vecs[j,2] * s
                    vecs[j,1] *= s
                    vecs[j,2] = s
                    vecs[j] += origin 
                out[i,j,:] = np.transpose(vecs[j])
            #end for j in [0 : n_samples)
        elif label == 3: #yz quad
            center = [0.5,0.5,0.5]
            vecs = np.random.random((n_samples, 3))
            for j in range(n_samples):
                vecs[j,0] = center[0]
                out[i,j,:] = np.transpose(vecs[j])
        elif label == 4: #xz quad
            center = [0.5,0.5,0.5]
            vecs = np.random.random((n_samples, 3))
            for j in range(n_samples):
                vecs[j,1] = center[1]
                out[i,j,:] = np.transpose(vecs[j])
        else: #random line trough the center
            point = [0.5,0.5,0.5]
            dir = np.random.random(3)
            dir = dir / np.linalg.norm(dir)
            point -= 0.5 * dir
            for j in range(n_samples):
                offset = np.random.random(1)
                sample = point + offset * dir
                out[i,j,:] =  np.transpose(sample)                                                                              
            #end for j in [0 : n_samples)
        labels[i] = label

        ###########################################
        #obj dump
        #mtl_file.write("newmtl id_" + str(i) + "\n")
        #mtl_file.write("illum 2\n")
        #mtl_file.write("Ka 0 0 0\n")
        #mtl_file.write("Kd " + str(np.random.random(1)[0]) + " " + str(np.random.random(1)[0]) + " " + str(np.random.random(1)[0]) + "\n")
        #mtl_file.write("Ks 0 0 0\n")
        #mtl_file.write("Ke 0 0 0\n")
        #mtl_file.write("Ni 1.0\n")
        #mtl_file.write("#end material " + str(i) + "\n")        

        #for j in range(n_samples):
        #    point_file.write("v " + str(out[i,0,j]) + " " + str(out[i,1,j]) + " " + str(out[i,2,j]) + "\n")
        
        #point_file.write("o " + str(i) + "\n" )
        #point_file.write("usemtl id_" + str(i) + "\n")
        #for j in range(n_samples):
        #    point_file.write("p " + str(i * n_samples + j + 1) + "\n")
        ###########################################
    #end for i in 0:batch_size

    ###########################################
    #obj dump
    #point_file.close()
    #mtl_file.close()
    ###########################################
    
    return np.expand_dims(out, axis=3), to_categorical(labels, num_labels)

import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
import os

if segmentation:
    point_data = Input(shape=(num_train_file, point_num, 3, 1), dtype='float32', name='point_cloud')
    # transformation 1
    out1 = Conv2D(64, (1, 3), padding='valid', batch_input_shape=(32, point_num, 3, 1), activation='relu')(point_data)
    out2 = Conv2D(128, (1, 1), activation='relu')(out1)
    out3 = Conv2D(128, (1, 1), activation='relu')(out2)

    # transformation 2
    out4 = Conv2D(512, (1, 1), activation='relu')(out3)
    out5 = Conv2D(2048, (1, 1), activation='relu')(out4)

    out_max = MaxPooling2D(pool_size=(point_num, 1))(out5)

    net = Flatten()(out_max)
    net = Dense(256, activation='relu')(net)
    net = Dense(256, activation='relu')(net)
    # net = Dropout(0.7)(net)
    cls_out = Dense(num_labels, activation='softmax')(net)

    # segmentation network
    label = Input(shape=(num_train_file, 1, 1, num_labels))
    out_max = keras.layers.concatenate([out_max, label], axis = 3)

    expand = keras.backend.tile(out_max, [1, point_num, 1, 1])
    concat = keras.layers.concatenate([expand, out1, out2, out3, out4, out5], axis = 3)


    net2 = Conv2D(256, (1,1), padding='valid', activation='relu')(concat)
    net2 = Dropout(0.8)(net2)
    net2 = Conv2D(256, (1,1), padding='valid', activation='relu')(net2)
    net2 = Dropout(0.8)(net2)
    net2 = Conv2D(128, (1,1), padding='valid', activation='relu')(net2)
    net2 = Conv2D(part_num, (1,1), padding='valid')(net2)

    net2 = tf.reshape(net2, [batch_size, num_point, part_num])

else:
    model = Sequential()
    model.add(Conv2D(64, (1, 3), padding='valid', batch_input_shape=(32, point_num, 3, 1))) #Note: assumes tensor flow channel ordering B x W x H x C
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (1, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(1024, (1, 1)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(point_num, 1)))
    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))


#sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#data,labels = generate_data(32, 1024)

#data, labels = generate_data(4096, num_point)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
data_dir = os.path.join(BASE_DIR, './scenes')

TRAINING_FILE_LIST = os.path.join(data_dir, 'train_file_list_large.txt')
TESTING_FILE_LIST = os.path.join(data_dir, 'test_file_list_large.txt')

train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
num_train_file = len(train_file_list)
test_file_list = provider.getDataFiles(TESTING_FILE_LIST)
num_test_file = len(test_file_list)

train_data = np.empty([num_train_file, point_num, 3], dtype=np.float32)
train_labels = np.empty([num_train_file], dtype=np.int32)

test_data = np.empty([num_test_file, point_num, 3], dtype=np.float32)
test_labels = np.empty([num_test_file], dtype=np.int32)

for i in range(num_train_file):
    cur_train_filename = os.path.join(data_dir, train_file_list[i])
    data, label, _ = provider.loadDataFile_with_seg(cur_train_filename, point_num)

    train_data[i, :, :] = data
    train_labels[i] = label

model.fit(np.expand_dims(train_data, axis = 3), to_categorical(train_labels, num_labels), batch_size=32, epochs=30)

print('Testing:')
#test_data, test_labels = generate_data(32, num_point)
for i in range(num_test_file):
    cur_test_filename = os.path.join(data_dir, test_file_list[i])
    data, label, _ = provider.loadDataFile_with_seg(cur_test_filename, point_num)

    test_data[i, :, :] = data
    test_labels[i] = label

score = model.evaluate(np.expand_dims(test_data, axis = 3), to_categorical(test_labels, num_labels), batch_size=32)
print('Final score: ', score)
