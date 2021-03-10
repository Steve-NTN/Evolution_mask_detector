from numpy.random import randint
from random import choice
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def encode_to_gen(n, size_gen):
    binary = bin(n)[2:]
    gen = [0] * size_gen
    start_gen = size_gen - len(binary)
    gen[start_gen:] = [int(x) for x in binary]
    return gen

def decode_gen(gen):
    return int('0b' + ''.join(str(i) for i in gen), 2)

class Individual:
    def __init__(self):
        self.epoch = randint(0, 4)                       # epoch
        self.input_size = randint(0, 4)                  # input size
        self.batch_size = randint(0, 4)                  # batch size
        # conv 1st
        self.filter_size_1 = randint(0, 8)               # filter size 1
        self.kernel_size_1 = randint(0, 2)               # kernel size 1
        self.pooling_1 = randint(0, 2)                   # type pooling 1
        self.pooling_size_1 = randint(0, 2)              # pooling size 1
        self.activation_1 = randint(0, 2)                # type activation 1
        self.padding_1 = randint(0, 2)                   # type padding conv 1
        # conv 2nd
        self.filter_size_2 = randint(0, 8)               # filter size 2
        self.kernel_size_2 = randint(0, 2)               # kernel size 2
        self.pooling_2 = randint(0, 2)                   # type pooling 2
        self.pooling_size_2 = randint(0, 2)              # pooling size 2
        self.activation_2 = randint(0, 2)                # type activation 2
        self.padding_2 = randint(0, 2)                   # type padding conv 2
        # conv 3th
        self.filter_size_3 = randint(0, 8)               # filter size 3
        self.kernel_size_3 = randint(0, 2)               # kernel size 3
        self.pooling_3 = randint(0, 2)                   # type pooling 3
        self.pooling_size_3 = randint(0, 2)              # pooling size 3
        self.activation_3 = randint(0, 2)                # type activation 3
        self.padding_3 = randint(0, 2)                   # type padding conv 3

        self.activation_4 = randint(0, 2)                # type activation 4
        self.activation_5 = 'softmax'                    # type activation 5
        self.dropout_1 = randint(0, 4)                   # dropout size 1
        self.dense_1 = randint(0, 8)                     # dense size 1
        self.loss_func = randint(0, 2)                   # loss function
        self.optimizer = randint(0, 2)                   # optimization type
        self.learn_rate = randint(0, 4)                  # learn rate
        
    
    # Get information about individal gen
    def individual_info(self, gen_array):
        print('_' * 50 + '\n{}'.format(gen_array))
        print('epoch: {}, input: {}, batch: {}'.format(gen_array[:2], gen_array[2:4], gen_array[4:6]))
        for i in range(3):
            print('conv {}: filter: {}, kernel: {}, activ: {}, pool: {}, pool_size: {}, padding: {}'.format(
                i+1, gen_array[6+i*8:9+i*8], gen_array[9+i*8:10+i*8], gen_array[10+i*8:11+i*8],
                gen_array[11+i*8:12+i*8], gen_array[12+i*8:13+i*8], gen_array[13+i*8:14+i*8]
            ))
        print('dropout: {}, dense: {}, activ: {}, loss: {}, opti: {}, learn: {}'.format(
            gen_array[30:32], gen_array[32:35], gen_array[35:36], gen_array[36:37],
            gen_array[37:38], gen_array[38: 40]
        ))
        print('_' * 50)

    # Get binary gen
    def individual_binary(self):
        # init individual with 40 parameter
        indiv = np.array([0]*40)

        # ______6 bit______|_____________8 bit x 3 = 24 bit____________|_________________10 bit__________________
        # epoch|input|batch|filter|kernel|activ|pool |pool_size|padding|dropout|dense|activ|loss_f|optim|learn_r| 
        # 2 bit|2 bit|2 bit|3 bit |1 bit |1 bit|1 bit|  1 bit  | 1 bit | 2 bit |3 bit|1 bit|1 bit |1 bit| 2 bit |

        bit = [2, 2, 2] + [3, 1, 1, 1, 1, 1] * 3 + [2, 3, 1, 1, 1, 2]
        param = [self.epoch, self.input_size, self.batch_size, self.filter_size_1, self.kernel_size_1,
                 self.pooling_1, self.pooling_size_1, self.activation_1, self.padding_1, self.filter_size_2, 
                 self.kernel_size_2, self.pooling_2, self.pooling_size_2, self.activation_2, self.padding_2, 
                 self.filter_size_3, self.kernel_size_3, self.pooling_3, self.pooling_size_3, self.activation_3, 
                 self.padding_3, self.dropout_1, self.dense_1, self.activation_4, self.loss_func, self.optimizer, 
                 self.learn_rate]
        start = 0
        for i in range(len(param)):
            indiv[start:start+bit[i]] = encode_to_gen(param[i], bit[i])
            start += bit[i]
        
        return indiv

    # Get model 
    def individual_model(self, p, num_class, x_train, y_train, x_test, y_test): 
        input_shape=(p['input'], p['input'], 3)

        model = Sequential()
        # conv 1st
        model.add(layer=Conv2D(p['filter_1'], p['kernel_1'], padding=p['padding_1'], activation=p['activation_1'], input_shape=(150, 150, 3)))
        if p['pooling_1'] == 'max':
            model.add(layer=MaxPooling2D(p['pooling_size_1'][0]))
        elif p['pooling_1'] == 'average':
            model.add(layer=AveragePooling2D(p['pooling_size_1'][0]))
        # conv 2nd
        model.add(layer=Conv2D(p['filter_2'], p['kernel_2'], padding=p['padding_2'], activation=p['activation_2']))
        if p['pooling_2'] == 'max':
            model.add(layer=MaxPooling2D(p['pooling_size_2'][0]))
        elif p['pooling_2'] == 'average':
            model.add(layer=AveragePooling2D(p['pooling_size_2'][0]))
        # conv 3th
        model.add(layer=Conv2D(p['filter_3'], p['kernel_3'], padding=p['padding_3'], activation=p['activation_3']))
        if p['pooling_3'] == 'max':
            model.add(layer=MaxPooling2D(p['pooling_size_3'][0]))
        elif p['pooling_3'] == 'average':
            model.add(layer=AveragePooling2D(p['pooling_size_3'][0]))
        # dense layer
        model.add(layer=Flatten())
        model.add(layer=Dropout(p['dropout_1']))
        model.add(layer=Dense(p['dense_1'], activation=p['activation_4']))
        model.add(layer=Dense(num_class, activation=p['activation_5']))

        # plot_model(model, to_file='images/model_plot_{}.png'.format(index_p), show_shapes=True, show_layer_names=True)
        # choose type to optimizer
        if p['optimizer'] == 'adam':
            opt = Adam(lr=p['learn_rate'], decay=p['learn_rate']/ p['epoch'])
        else:
            opt = Adamax(lr=p['learn_rate'], decay=p['learn_rate']/ p['epoch'])
        model.compile(optimizer=opt, loss=p['loss_func'], metrics=['accuracy'])
        
        # construct the training image generator for data augmentation
        aug = ImageDataGenerator(
            rotation_range=40,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest")
        # train the head of the network
        print("[INFO] training head...")
        H = model.fit(
            aug.flow(x_train, y_train, batch_size=p['batch']),
            steps_per_epoch=len(x_train) // p['batch'],
            validation_data=(x_test, y_test),
            validation_steps=len(x_test) // p['batch'],
            epochs=p['epoch'])
        return model

    def individual_decode(self, binary_gen):
        input_size = [70, 120, 170, 220]
        batch = [10, 15, 20, 25]
        kernel_size = [(3, 3), (5, 5)]
        pooling = ['max', 'average']
        pooling_size = [(2, 2), (3, 3)]
        activation = ['sigmoid', 'relu']
        padding = ['valid', 'same']
        dropout = [0.1, 0.2, 0.3, 0.4]
        loss_func = ['categorical_crossentropy', 'binary_crossentropy']
        learn_rate = [1e-2, 1e-3, 1e-4, 1e-5]
        optimizer = ['adamax', 'adam']

        params = {'epoch': decode_gen(binary_gen[:2]) + 5,                             
                'input': input_size[decode_gen(binary_gen[2:4])],                 
                'batch': batch[decode_gen(binary_gen[4:6])],                      
                # conv 1st (8 bit)
                'filter_1': decode_gen(binary_gen[6:9]) + 30,                 
                'kernel_1': kernel_size[decode_gen(binary_gen[9:10])],          
                'pooling_1': pooling[decode_gen(binary_gen[10:11])],                 
                'pooling_size_1': pooling_size[decode_gen(binary_gen[11:12])],  
                'activation_1': activation[decode_gen(binary_gen[12:13])],        
                'padding_1': padding[decode_gen(binary_gen[13:14])],               
                # conv 2nd (8 bit)
                'filter_2': decode_gen(binary_gen[14:17]) + 60,
                'kernel_2': kernel_size[decode_gen(binary_gen[17:18])],
                'pooling_2': pooling[decode_gen(binary_gen[18:19])],
                'pooling_size_2': pooling_size[decode_gen(binary_gen[19:20])],
                'activation_2': activation[decode_gen(binary_gen[20:21])],
                'padding_2': padding[decode_gen(binary_gen[21:22])],
                # conv 3th (8 bit)
                'filter_3': decode_gen(binary_gen[22:25]) + 120,
                'kernel_3': kernel_size[decode_gen(binary_gen[25:26])],
                'pooling_3': pooling[decode_gen(binary_gen[26:27])],
                'pooling_size_3': pooling_size[decode_gen(binary_gen[27:28])],
                'activation_3': activation[decode_gen(binary_gen[28:29])],
                'padding_3': padding[decode_gen(binary_gen[29:30])],
                # dense 1
                # 2 bit
                'dropout_1': dropout[decode_gen(binary_gen[30:32])],
                # 3 bit
                'dense_1': decode_gen(binary_gen[32:35]) + 250,
                # 1 bit
                'activation_4': activation[decode_gen(binary_gen[35:36])],
                'activation_5': 'softmax',
                # loss function, optimizer & learn rate
                # 1 bit
                'loss_func': loss_func[decode_gen(binary_gen[36:37])],
                # 1 bit
                'optimizer': optimizer[decode_gen(binary_gen[37:38])],
                # 2 bit
                'learn_rate': learn_rate[decode_gen(binary_gen[38:40])] 
                }
        return params
        