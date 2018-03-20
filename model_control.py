#! -*- coding:utf-8 -*-
'''

Control logic module

@author: LouisZBravo

'''

import data_util as du
from model import ConvQAModelGraph
import config as cfg
import sys
import os
ext_tool_dir = cfg.DirConfig.EXT_TOOL_DIR
sys.path.append(os.path.abspath(ext_tool_dir))
from evaluation import eval_in_model
from shutil import copyfile
from data_util import write_log
from keras.models import Model
import time

def exec_(param):
    
    '''initialize vocab for specified problem'''
    # common config and user control parameter loading
    qa_data_mode = param['data_mode'].qa_data
    wdim = cfg.ModelConfig.WORD_DIM_DICT[qa_data_mode]
    # use_preprocessed = param['model_mode'].use_preprocessed
    train_wv = param["preprocess_mode"].train_wv
    # wdim = cfg.ModelConfig.WORD_DIM[qa_data_mode]
    corpus_mode = cfg.PreProcessConfig.CORPUS_MODE
    ling_unit = cfg.PreProcessConfig.LING_UNIT # linguistic unit mode
    assert ling_unit in ("WORD", "CHAR")
    s_w_rmvl = cfg.PreProcessConfig.STOP_WORD_REMOVAL # stop-word removal mode 
    
    train_set = param['model_mode'].train_set
    eval_set = param['model_mode'].eval_set
    use_saved_4_training = param['model_mode'].use_saved_4_training
    use_saved_4_testing = param['model_mode'].use_saved_4_testing
    
    wv_path, qa_data_path_t, qa_data_path_e, model_weight_path, score_path = generate_model_paths(qa_data_mode, ling_unit, s_w_rmvl, train_set, eval_set)
    
    vocab = du.Vocab(wv_path)
    stop_set = set([])
    # initialize a vocab instance
    if train_wv: # train new vectors using corpus
        # clean the corpus text
        corpus_path = cfg.DirConfig.CORPUS_PATH_DICT[corpus_mode]
        clean_corpus_path = cfg.DirConfig.CLEAN_CORPUS_PATH_DICT[corpus_mode]
        
        print("---starting to clean corpus text at running time %f---" % time.clock())
        write_log("started to clean corpus text at running time %f\n" % time.clock())
        text_cleaner = du.TextCleaner(corpus_path)
        # load stop word list into text cleaner if needed
        if s_w_rmvl:
            stop_set = vocab.get_stop_word_set()
        
        text_cleaner.clean_chn_corpus_2file(clean_corpus_path, stop_set)
        
        # copy cleaned text to word2vec directory
        clean_corpus_filename = cfg.DirConfig.CLEAN_CORPUS_FILENAME_DICT[corpus_mode]
        dst_corpus_path = cfg.DirConfig.W2V_DIR + clean_corpus_filename
        copyfile(clean_corpus_path, dst_corpus_path)
        
        print("---starting to build vocab database at running time %f---" % time.clock())
        write_log("started to build vocab database at running time %f\n" % time.clock())
        vocab.build_vocab_database(qa_data_mode, clean_corpus_filename)
    else:
        print("---starting to load vocab from database at running time %f---" % time.clock())
        write_log("started to load vocab from database at running time %f\n" % time.clock())
        vocab.load_wv_from_db(qa_data_mode)
    
    '''
     initialize data stream for model
     data stream initiates unknown-word mapping
     so beware of deadlock in database operation
    '''
    if train_set != 'NAH': # training
        print("---starting to prepare training data stream at running time %f---" % time.clock())
        write_log("started to prepare training data stream at running time %f\n" % time.clock())
        data_stream = du.SentenceDataStream(qa_data_path_t, vocab, (qa_data_mode, 't'))
        
        print("---starting to initialize model for training at running time %f---" % time.clock())
        write_log("started to initialize model for training at running time %f\n" % time.clock())
        model_graph = ConvQAModelGraph(wdim)
        model_in = model_graph.get_model_inputs()
        model_out = model_graph.get_model_outputs()
        my_model = Model(inputs=model_in, outputs=model_out)
        
        # some hyper-parameters
        batch_size = data_stream.get_batch_size()
        train_epoch = cfg.ModelConfig.TRAIN_EPOCH
        loss_func = cfg.ModelConfig.LOSS_FUNC
        optm = cfg.ModelConfig.OPT
        # model initialization 
        my_model.compile(optimizer=optm, loss=loss_func, metrics=['accuracy'])
        if use_saved_4_training:
            try:
                my_model.load_weights(model_weight_path)
            except Exception as e:
                print("%s" % e)
                write_log("%s" % e)
        # start training
        print("---starting to feed model at running time %f---" % time.clock())
        write_log("started to feed model at running time %f\n" % time.clock())
        for _ in range(train_epoch):
            g = data_stream.get_batch()
            while(True):
                try:
                    q_batch, a_batch, label_batch, add_feat_batch = next(g)
                except StopIteration:
                    break
                x = [q_batch, a_batch, add_feat_batch]
                y = [label_batch]
                my_model.fit(x, y, batch_size=batch_size)
        my_model.save_weights(model_weight_path)
    
    if eval_set != 'NAH': # evaluation
        print("---starting to prepare evaluation data stream at running time %f---" % time.clock())
        write_log("started to prepare evaluation data stream at running time %f\n" % time.clock())
        
        predicted_score_lst = []
        score_file = open(score_path, 'wb')
        
        data_stream = du.SentenceDataStream(qa_data_path_e, vocab, (qa_data_mode, 'e'))
        print("---starting to initialize model for evaluation at running time %f---" % time.clock())
        write_log("started to initialize model for evaluation at running time %f\n" % time.clock())
        model_graph = ConvQAModelGraph(wdim)
        model_in = model_graph.get_model_inputs()
        model_out = model_graph.get_model_outputs()
        my_model = Model(inputs=model_in, outputs=model_out)
        loss_func = cfg.ModelConfig.LOSS_FUNC
        optm = cfg.ModelConfig.OPT
        my_model.compile(optimizer=optm, loss=loss_func, metrics=['accuracy'])
        if use_saved_4_testing:
            try:
                my_model.load_weights(model_weight_path)
            except Exception as e:
                print("%s" % e)
                write_log("%s" % e)
                
        # some hyper-parameters
        batch_size = data_stream.get_batch_size()
        g = data_stream.get_batch()
        while(True):
            try:
                q_batch, a_batch, add_feat_batch = next(g)
            except StopIteration:
                break
            x = [q_batch, a_batch, add_feat_batch]
            predicted_batch = list(my_model.predict(x, batch_size))
            predicted_score_lst.extend(predicted_batch)
            # y = model.predict(x, batch_size=batch_size)
        for sc in predicted_score_lst:
            score_to_write = (str(sc[0]) + '\n').encode('utf-8')
            score_file.write(score_to_write)
        score_file.close()
        res = eval_in_model(qa_data_path_e, score_path, '')
        write_log(res + '\n')
    
    write_log("Finished at time %f.\n" % time.clock())
    
def generate_model_paths(qa_data_mode, ling_unit, s_w_rmvl, train_set, eval_set):
    '''
     generate a tuple of paths of the files that the qa matching model needs
     including vector database path, training data path, evaluation data path
    pass
    '''
    
    """vector db path"""
    _data_dir = cfg.DirConfig.DATA_CACHED_DIR
    _wv_file_sfx = cfg.DirConfig.WV_FILE_SUFFIX
    if ling_unit == "WORD": # use parsed word as atomic representation
        if s_w_rmvl:
            _l_u = '_nonstop_w'
        else:
            _l_u = '_w'
    else: #CHAR mode
        _l_u = '_c'
    # relative path DATA_DIR + 'HITNLP' + ''_nonstop_w'_w'/'_c' + 'vec.db'
    # since sqlite3 tool only accepts abs path, we use abs path here rather than relative path
    _wv_re_path = _data_dir + qa_data_mode + _l_u + _wv_file_sfx
    wv_path = os.path.abspath(_wv_re_path)
    
    """training&evaluation data path"""
    qa_data_path_t = cfg.DirConfig.QA_DATA_PATH_DICT[qa_data_mode][train_set]
    qa_data_path_e = cfg.DirConfig.QA_DATA_PATH_DICT[qa_data_mode][eval_set]
    
    model_weight_path = cfg.DirConfig.MODEL_WEIGHTS_DIR
    score_path = cfg.DirConfig.PREDICTED_SCORE_DIR
    
    return (wv_path, qa_data_path_t, qa_data_path_e, model_weight_path, score_path)



def generate_corpus_path():
    '''
     generate a tuple of paths of the files that is used in word vector training
     including corpus path, cleaned corpus path
     seems not needed.............
    '''
    pass