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
    write_log("--------------------------------------------------------\n")
    
    global time_point
    time_point = time.clock()
    # common config and user control parameter loading
    qa_data_mode = param['data_mode'].qa_data
    wdim = cfg.ModelConfig.WORD_DIM_DICT[qa_data_mode]
    # use_preprocessed = param['model_mode'].use_preprocessed
    train_wv = param["preprocess_mode"].train_wv
    clean_corpus = param["preprocess_mode"].clean_corpus
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
    # initialize a vocab instance
    if train_wv: # train new vectors using corpus
        # clean the corpus text
        corpus_path = cfg.DirConfig.CORPUS_PATH_DICT[corpus_mode]
        clean_corpus_path = cfg.DirConfig.CLEAN_CORPUS_PATH_DICT[corpus_mode]
        
        if clean_corpus:
            print("---starting to clean corpus text at running time %f---" % (time.clock() - time_point))
            write_log("started to clean corpus text at running time %f\n" % (time.clock() - time_point))
            text_cleaner = du.TextCleaner(corpus_path)
            
            text_cleaner.clean_chn_corpus_2file(clean_corpus_path)
        
        # copy cleaned text to word2vec directory
        clean_corpus_filename = cfg.DirConfig.CLEAN_CORPUS_FILENAME_DICT[corpus_mode]
        dst_corpus_path = cfg.DirConfig.W2V_DIR + clean_corpus_filename
        copyfile(clean_corpus_path, dst_corpus_path)
        
        print("---starting to build vocab database at running time %f---" % (time.clock() - time_point))
        write_log("started to build vocab database at running time %f\n" % (time.clock() - time_point))
        vocab.build_vocab_database(qa_data_mode, clean_corpus_filename)
    else:
        print("---starting to load vocab from database at running time %f---" % (time.clock() - time_point))
        write_log("started to load vocab from database at running time %f\n" % (time.clock() - time_point))
        vocab.load_wv_from_db(qa_data_mode)
    
    
    '''
     initialize data stream for model and start training
     (canceled) data stream initiates unknown-word mapping
     (canceled) so beware of deadlock in database operation
    '''
    if train_set != 'NAH' and eval_set != 'NAH': # do one time evaluation before training for comparison
        print("evaluation before training:")
        write_log("evaluation before training:\n")
        print("---starting to prepare evaluation data stream at running time %f---" % (time.clock() - time_point))
        write_log("started to prepare evaluation data stream at running time %f\n" % (time.clock() - time_point))
        
        predicted_score_lst = []
        score_file = open(score_path, 'wb')
        
        data_stream = du.SentenceDataStream(qa_data_path_e, vocab, (qa_data_mode, 'e'))
        print("---starting to initialize model for evaluation at running time %f---" % (time.clock() - time_point))
        write_log("started to initialize model for evaluation at running time %f\n" % (time.clock() - time_point))
        model_graph = ConvQAModelGraph(wdim)
        model_in = model_graph.get_model_inputs()
        model_out = model_graph.get_model_outputs()
        my_model = Model(inputs=model_in, outputs=model_out)
        loss_func = cfg.ModelConfig.LOSS_FUNC
        optm = cfg.ModelConfig.OPT
        my_model.compile(optimizer=optm, loss=loss_func, metrics=['accuracy'])
        if use_saved_4_training: # in consistency with training here
            try:
                my_model.load_weights(model_weight_path)
            except Exception as e:
                print("%s" % e)
                write_log("%s" % e)
        else:
            pass
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
        write_log("before-training evaluation ends\n")
        write_log("----------------------------\n")
        
    
    if train_set != 'NAH': # training
        print("---starting to prepare training data stream at running time %f---" % (time.clock() - time_point))
        write_log("started to prepare training data stream at running time %f\n" % (time.clock() - time_point))
        data_stream = du.SentenceDataStream(qa_data_path_t, vocab, (qa_data_mode, 't'))
        
        print("---starting to initialize model for training at running time %f---" % (time.clock() - time_point))
        write_log("started to initialize model for training at running time %f\n" % (time.clock() - time_point))
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
        else:
            pass
            
        # start training
        print("---starting to feed model at running time %f---" % (time.clock() - time_point))
        write_log("started to feed model at running time %f\n" % (time.clock() - time_point))
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
        
        print(vocab._unk_num) #######################################
        print(len(vocab.kn_set))
        write_log("In %s set, %d known words and %d unknown words found.\n" % (eval_set, len(vocab.kn_set), vocab._unk_num))
        
    if eval_set != 'NAH': # evaluation
        print("---starting to prepare evaluation data stream at running time %f---" % (time.clock() - time_point))
        write_log("started to prepare evaluation data stream at running time %f\n" % (time.clock() - time_point))
        
        predicted_score_lst = []
        score_file = open(score_path, 'wb')
        
        data_stream = du.SentenceDataStream(qa_data_path_e, vocab, (qa_data_mode, 'e'))
        print("---starting to initialize model for evaluation at running time %f---" % (time.clock() - time_point))
        write_log("started to initialize model for evaluation at running time %f\n" % (time.clock() - time_point))
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
        else:
            pass
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
        
        print(vocab._unk_num) #######################################
        print(len(vocab.kn_set))
        write_log("In %s set, %d known words and %d unknown words found.\n" % (eval_set, len(vocab.kn_set), vocab._unk_num))
        
        res = eval_in_model(qa_data_path_e, score_path, '')
        write_log(res + '\n')
        
    
    write_log("Finished at time %f.\n" % (time.clock() - time_point))
    write_log("--------------------------------------------------------\n\n")
    time_point = time.clock()
    
def grid_search(param):
    h_param_lst = []
    for t_epch in (5, 10, 15, 20, 25):
        for b_size in (5, 10, 32, 50, 64, 80, 100):
            for feat_map_num in (50, 75, 100, 125, 150):
                for conv_ftr_len in (1, 2, 3, 4, 5, 6, 7, 8):
                    h_param_lst.append((feat_map_num, conv_ftr_len, b_size, t_epch))
    
    for cur_h_param in h_param_lst:
        write_log("\n\n")
        write_log("using hyper parameters:\n")
        write_log("LING_UNIT = %s\n" % str(cfg.PreProcessConfig.LING_UNIT))
        write_log("PUNCTUALATION_REMOVAL = %s\n" % str(cfg.PreProcessConfig.PUNCTUALATION_REMOVAL))
        write_log("STOP_WORD_REMOVAL = %s\n" % str(cfg.PreProcessConfig.STOP_WORD_REMOVAL))
        write_log("SORT_INSTANCE = %s\n" % str(cfg.ModelConfig.SORT_INSTANCE))
        write_log("PAD_WIDE = %s\n" % str(cfg.ModelConfig.PAD_WIDE))
        write_log("----------------------------\n")
        
        write_log("FEATURE_MAP_NUM = %d\n" % cur_h_param[0])
        write_log("CONV_FILTER_LEN = %d\n" % cur_h_param[1])
        write_log("BATCH_SIZE = %d\n" % cur_h_param[2])
        write_log("TRAIN_EPOCH = %d\n" % cur_h_param[3])
        cfg.ModelConfig.FEATURE_MAP_NUM = cur_h_param[0]
        cfg.ModelConfig.CONV_FILTER_LEN = cur_h_param[1]
        cfg.ModelConfig.BATCH_SIZE = cur_h_param[2]
        cfg.ModelConfig.TRAIN_EPOCH = cur_h_param[3]
    
        exec_(param)

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
        _u = '_w'
    else: #CHAR mode
        _u = '_c'
    # relative path DATA_DIR + 'HITNLP' + ''_nonstop_w'_w'/'_c' + 'vec.db'
    # since sqlite3 tool only accepts abs path, we use abs path here rather than relative path
    _wv_re_path = _data_dir + qa_data_mode + _u + _wv_file_sfx
    #_wv_nons_re_path = _data_dir + qa_data_mode + '_nonstop' + _u + _wv_file_sfx
    wv_path = os.path.abspath(_wv_re_path)
    #wv_ns_path = os.path.abspath(_wv_nons_re_path)
    
    """training&evaluation data path"""
    qa_data_path_t = cfg.DirConfig.QA_DATA_PATH_DICT[qa_data_mode][train_set]
    qa_data_path_e = cfg.DirConfig.QA_DATA_PATH_DICT[qa_data_mode][eval_set]
    
    model_weight_path = cfg.DirConfig.MODEL_WEIGHTS_DIR
    # ini_weight_path = cfg.DirConfig.MODEL_INITIALIZED_WEIGHTS_DIR
    score_path = cfg.DirConfig.PREDICTED_SCORE_DIR
    
    return (wv_path, qa_data_path_t, qa_data_path_e, model_weight_path, score_path)



def generate_corpus_path():
    '''
     generate a tuple of paths of the files that is used in word vector training
     including corpus path, cleaned corpus path
     seems not needed.............
    '''
    pass