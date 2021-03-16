from individual import decode_gen


def individual_decode(binary_gen):
    input_size = [70, 120, 170, 220]
    batch = [10, 15, 20, 25]
    filter = [16, 32, 64, 86, 128, 172, 218, 256]
    kernel_size = [(3, 3), (5, 5)]
    pooling = ['max', 'averg']
    pooling_size = [(2, 2), (3, 3)]
    activation = ['sigmoi', 'relu']
    padding = ['valid', 'same']
    dropout = [0.1, 0.2, 0.3, 0.4]
    loss_func = ['categor', 'binary']
    learn_rate = [1e-2, 1e-3, 1e-4, 1e-5]
    optimizer = ['adamax', 'adam']

    params = {'epoch': decode_gen(binary_gen[:3]) + 4,                             
            # 'input': input_size[decode_gen(binary_gen[2:4])],                 
            'batch': batch[decode_gen(binary_gen[3:6])],                      
            # conv 1st (8 bit)
            'filter_1': filter[decode_gen(binary_gen[6:9])],                 
            'kernel_1': kernel_size[decode_gen(binary_gen[9:10])],          
            'pooling_1': pooling[decode_gen(binary_gen[10:11])],                 
            'pooling_size_1': pooling_size[decode_gen(binary_gen[11:12])],  
            'activation_1': activation[decode_gen(binary_gen[12:13])],        
            'padding_1': padding[decode_gen(binary_gen[13:14])],               
            # conv 2nd (8 bit)
            'filter_2': filter[decode_gen(binary_gen[14:17])],
            'kernel_2': kernel_size[decode_gen(binary_gen[17:18])],
            'pooling_2': pooling[decode_gen(binary_gen[18:19])],
            'pooling_size_2': pooling_size[decode_gen(binary_gen[19:20])],
            'activation_2': activation[decode_gen(binary_gen[20:21])],
            'padding_2': padding[decode_gen(binary_gen[21:22])],
            # conv 3th (8 bit)
            'filter_3': filter[decode_gen(binary_gen[22:25])],
            'kernel_3': kernel_size[decode_gen(binary_gen[25:26])],
            'pooling_3': pooling[decode_gen(binary_gen[26:27])],
            'pooling_size_3': pooling_size[decode_gen(binary_gen[27:28])],
            'activation_3': activation[decode_gen(binary_gen[28:29])],
            'padding_3': padding[decode_gen(binary_gen[29:30])],
            # dense 1
            # 2 bit
            'dropout_1': dropout[decode_gen(binary_gen[30:32])],
            # 3 bit
            'dense_1': filter[decode_gen(binary_gen[32:35])],
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

def print_list_indiv(list_indi):

    # Header
    print('_' * 100)
    print('#' + ' ' *3 + 'Fitness' + ' ' * 3 + 'Architecture' + ' ' *50) 
    print('_' * 100)

    list_indi = sorted(list_indi, key=lambda tup: tup[1], reverse=True)

    for ind in range(len(list_indi)):
        param = individual_decode(list_indi[ind][0])
        column = 0
        text = ''
        for key, value in param.items():
            if column == 6: 
                column = 0
                text += '\n' + ' ' * 14
            else:
                text_col = key + '=' + str(value)
                text += text_col + ' '* (15 - len(text_col)) 
                column += 1

        print('{}{}{}{}{}'.format(ind + 1, ' ' * 3, round(list_indi[ind][1], 5), ' ' * 3, text))
        print('_' * 100)


if __name__ == "__main__":
    list_indi = [([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
       1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1], 0.9291819334030151), ([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0,
       1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1], 0.9426129460334778), ([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0,
       1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1], 0.9597069621086121), ([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
       1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1], 0.9242979288101196), ([1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
       1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1], 0.930402934551239)]
    
    print_list_indiv(list_indi)