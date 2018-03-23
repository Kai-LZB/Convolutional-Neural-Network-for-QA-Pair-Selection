#! -*- coding:utf-8 -*-
'''

Interface for programmer

@author: LouisZBravo

'''

from model_control import exec_, grid_search

class DataMode():
    '''
     configuration for problem to solve
     mode option:
         qa_data:
             "TREC": TREC QA data for model, ?? for corpus
             "HITNLP": QA data from NLP class 17 summer, vocab from People's Daily for corpus
             
    '''
    def __init__(self):
        self.qa_data = "HITNLP" # 'HITNLP', 'ENG_TEST', 'CHN_TEST'
        

class PreprocessMode():
    '''
     determine whether to train word vectors
     mode option:
         train_wv:
             True: train word vectors using external tool and save to database
             False: using existing word vectors
         wv_src:
             'ENG_TEST', 'CHN_TEST', 'PEOPLE_S_DAILY'
             only matters when train_wv is True
             
             
         non_stop should go to config!!!!!
             "HITNLP_nonstop" QA data from NLP class 17 summer, non-stop vocab from People's Daily for corpus
         unit should go to config!!!!!!
             "WORD": using word as linguistic vector unit
             "CHAR": using character as linguistic vector unit
         pad? should go to config
         
    '''
    def __init__(self):
        self.train_wv = True # decides whether to train new vectors
        self.clean_corpus = True

class ModelMode():
    '''
     configuration for model's behavior:
         preprocess -> training -> evaluating
     mode option:
         use_preprocessed:
             True: use existing preprocessed data
             False: preprocess data again
         use_trained:
             True: load existing trained model
             False: initialize a new untrained model
         train_set: 
             "DEV": train on developing data set
             "TRAIN": train on training set
             "NAH": do no training
         eval_set:
             "DEV": generate test result based on developing data set and save result data for later use
             "TEST": generate test result based on testing data set and save result data for later use
             "TRAIN": generate test result based on training data set and save result data for later use
             "NAH": do no evaluation
    '''
    def __init__(self):
        # most commonly used setting
        self.use_saved_4_training = False
        self.use_saved_4_testing = False
        self.train_set = "NAH"
        self.eval_set = "DEV"
    

if __name__ == '__main__':
    to_grid_search = False # if True: search the hyperparam space and train accordingly; if False: use fixed value of config.py
    param = {}
    data_mode = DataMode()
    preprocess_mode = PreprocessMode()
    model_mode = ModelMode()
    '''
     ...code for interaction and setting mode parameters...
    '''
    
    param["data_mode"] = data_mode
    param["preprocess_mode"] = preprocess_mode
    param["model_mode"] = model_mode
    
    if to_grid_search:
        grid_search(param)
    else:
        exec_(param)
    