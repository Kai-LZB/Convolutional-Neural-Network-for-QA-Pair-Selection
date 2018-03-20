#! -*- coding:utf-8 -*-
'''

Model graph module

@author: LouisZBravo

'''
import config as cfg
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dot, Concatenate, Dense, Dropout
from keras import regularizers

class ConvQAModelGraph(object):
    def __init__(self, wdim):
        # hyperparameters
        conv_filter_len = cfg.ModelConfig.CONV_FILTER_LEN
        feat_map_num = cfg.ModelConfig.FEATURE_MAP_NUM
        
        # input dim: (sentence length, word dim)
        _q_input = Input(shape=(None, wdim))
        _a_input = Input(shape=(None, wdim))
        _add_feat_input = Input(shape=(4,))
        self.model_inputs = (_q_input, _a_input, _add_feat_input)
        
        # feature map dim: (sent_len-filter_len+1, feat_map_num)
        _q_feature_maps = Conv1D(input_shape = (None, wdim),
                                 filters = feat_map_num,
                                 kernel_size = conv_filter_len,
                                 activation = 'relu',
                                 kernel_regularizer = regularizers.l2(0.00001),
                                 )(_q_input)
        _a_feature_maps = Conv1D(input_shape = (None, wdim),
                                 filters = feat_map_num,
                                 kernel_size = conv_filter_len,
                                 activation = 'relu',
                                 kernel_regularizer = regularizers.l2(0.00001),
                                 )(_a_input)
                                 
        # pooling res dim: (feat_map_num, )
        _q_pooled_maps = GlobalMaxPooling1D()(_q_feature_maps)
        _a_pooled_maps = GlobalMaxPooling1D()(_a_feature_maps)
        
        # sentence match res dim: (1, )
        #sent_match_layer_0 = DotMatrixLayer(output_dim = feat_map_num)
        sent_match_layer_0 = Dense(units = feat_map_num,
                                   activation = None,
                                   use_bias = False,
                                   kernel_regularizer = regularizers.l2(0.0001),
                                   )
        sent_match_layer_1 = Dot(axes=-1)
        _qM_dot_res = sent_match_layer_0(_q_pooled_maps)
        _sent_match_res = sent_match_layer_1([_qM_dot_res, _a_pooled_maps])
        
        # concatenate res dim: (2*feat_map_num+5, )
        _conc_res = Concatenate()([_q_pooled_maps, _sent_match_res, _a_pooled_maps, _add_feat_input])
        
        # hidden layer out dim: (2*feat_map_num+5, )
        _hid_res = Dense(units = 2 * feat_map_num + 5,
                         activation = 'tanh',
                         use_bias = True,
                         kernel_regularizer = regularizers.l2(0.0001),
                         )(_conc_res)
                         
        # dropout some units before computing softmax result
        _dropped_hid_res = Dropout(rate=0.5)(_hid_res)
                        
        # softmax binary classifier out dim: (2, )
        """_bin_res = Dense(units = 2,
                         activation = 'softmax',
                         use_bias = False,
                         kernel_regularizer = regularizers.l2(0.0001),
                         )(_dropped_hid_res)
        
        self.model_output = _bin_res[:, 0]"""
        
        _res = Dense(units = 1,
                     activation = 'sigmoid',
                     use_bias = False,
                     kernel_regularizer = regularizers.l2(0.0001),
                     )(_dropped_hid_res)
        self.model_output = _res
        
    def get_model_inputs(self):
        return self.model_inputs
    
    def get_model_outputs(self):
        return self.model_output
    
        
        