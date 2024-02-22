from __future__ import absolute_import

from Networks.utils import *
from Networks.arch_parts import *
from Networks.act import *
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def unetpp_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, 
                 activation='ReLU', apply_batch_norm=False, pool=True, unpool=True, 
                 name='unetpp'):
    
    
    activation_func = eval(activation)

    inputs_skip = []
    depth_ = len(filter_num)


    inputs = input_tensor

    # stacked conv2d before downsampling
    inputs = convolutional_stack(inputs, filter_num[0], stack_num=stack_num_down, activation=activation, 
                   apply_batch_norm=apply_batch_norm, name='{}_down0'.format(name))
    inputs_skip.append(inputs)

    # downsampling blocks
    for i, f in enumerate(filter_num[1:]):
        inputs = base_left(inputs, f, stack_num=stack_num_down, activation=activation, pool=pool, 
                      apply_batch_norm=apply_batch_norm, name='{}_down{}'.format(name, i+1))        
        inputs_skip.append(inputs)

    # reverse indexing encoded feature maps
    inputs_skip = inputs_skip[::-1]
    # upsampling begins at the deepest available tensor
    inputs = inputs_skip[0]
    # other tensors are preserved for concatenation
    inputs_decode = inputs_skip[1:]
    depth_decode = len(inputs_decode)

    # reverse indexing filter numbers
    filter_num_decode = filter_num[:-1][::-1]

    # upsampling with concatenation
    for i in range(depth_decode):
        inputs = base_right(inputs, [inputs_decode[i],], filter_num_decode[i], stack_num=stack_num_up, activation=activation, 
                       unpool=unpool, apply_batch_norm=apply_batch_norm, name='{}_up{}'.format(name, i))

    # if tensors for concatenation is not enough
    # then use upsampling without concatenation 
    if depth_decode < depth_-1:
        for i in range(depth_-depth_decode-1):
            i_real = i + depth_decode
            inputs = base_right(inputs, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation, 
                       unpool=unpool, apply_batch_norm=apply_batch_norm, concat=False, name='{}_up{}'.format(name, i_real))   
    return inputs

def unetpp_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
            activation='ReLU', output_activation='Softmax', apply_batch_norm=False, pool=True, unpool=True, 
            name='unetpp'):
    
    activation_func = eval(activation)
        
    IN = Input(input_size)
    
    # base    
    inputs = unetpp_2d_base(IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up, 
                     activation=activation, apply_batch_norm=apply_batch_norm, pool=pool, unpool=unpool,  
                     name=name)
    
    # output layer
    OUT = convolutional_output(inputs, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    
    # functional API model
    model = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(name))
    
    return model