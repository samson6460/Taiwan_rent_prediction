# %%
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import os

# %%
class Data_reader():
    def __init__(self, path_name, mode='categorical', label_name = None):
        self.path_name = path_name
        self.mode = mode
        if label_name:
            self.label_name = label_name
        elif mode=='binary' or mode=='categorical':   
            self.label_name = [f[:(f.rfind('.'))] for f in os.listdir(path_name) if not f.startswith('.')]
            if len(self.label_name)>2 and mode=='binary':
                raise ValueError('Found more than two files, please assign parameter, label_name.')
            

    def read(self, maxlen=None, shuffle=True, random_seed=None, encoding='utf-8'):      
        train_data = []
        label_data = []

        if self.mode=='regression':
            f = open(self.path_name,encoding=encoding,errors='ignore')
            for d in f:
                try:
                    data = d.split(' ')
                    train_value = list(map(float,data[0].split(',')))
                    label_value = list(map(float,data[1].split(',')))
                    train_data.append(train_value)
                    label_data.append(label_value)
                except Exception as e:
                    print(e)
                    pass
            f.close()

            label_data = np.array(label_data)

        else:
            file_list = [f for f in os.listdir(self.path_name) if not f.startswith('.')]

            for name in file_list:
                f = open(self.path_name+os.sep+name,encoding=encoding,errors='ignore')
                label = self.label_name.index(name[:(name.rfind('.'))])
                for d in f:
                    try:
                        data = list(map(float,d.split(',')))
                        train_data.append(data)
                        label_data.append(label)
                    except Exception as e:
                        print(e)
                f.close()

            if self.mode=='categorical':
                label_data = to_categorical(label_data)
            elif self.mode=='binary':
                label_data = np.array(label_data)
            else:
                raise ValueError('invalid input of mode')

        if maxlen==None:
            train_data = np.array(train_data)  
        else:  
            train_data = pad_sequences(train_data, maxlen=maxlen, truncating='post').astype('float')

        if shuffle:
            np.random.seed(random_seed)
            shuffle_index = np.arange(len(train_data))
            np.random.shuffle(shuffle_index)
            train_data = train_data[shuffle_index]
            label_data = label_data[shuffle_index]

        return train_data, label_data
