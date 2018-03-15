#! -*- coding:utf-8 -*-
'''

Control logic module

@author: LouisZBravo

'''

import data_util as du
import config as cfg
import os
from shutil import copyfile
from data_util import write_log

def exec_(param):
    
    '''initialize vocab for specified problem'''
    # common config and user control parameter loading
    qa_data_mode = param['data_mode'].qa_data
    use_preprocessed = param['model_mode'].use_preprocessed
    train_wv = param["preprocess_mode"].train_wv
    # wdim = cfg.ModelConfig.WORD_DIM[qa_data_mode]
    corpus_mode = cfg.PreProcessConfig.CORPUS_MODE
    ling_unit = cfg.PreProcessConfig.LING_UNIT # linguistic unit mode
    assert ling_unit in ("WORD", "CHAR")
    s_w_rmvl = cfg.PreProcessConfig.STOP_WORD_REMOVAL # stop-word removal mode 
    # generate wv db path according to the problem to solve e.g. HITNLP problem
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
    train_set = param['model_mode'].train_set
    eval_set = param['model_mode'].eval_set
    qa_data_path_t = cfg.DirConfig.QA_DATA_PATH_DICT[qa_data_mode][train_set]
    qa_data_path_e = cfg.DirConfig.QA_DATA_PATH_DICT[qa_data_mode][eval_set]
    
    vocab = du.Vocab(wv_path)
    stop_set = set([])
    # decide whether to train new word embedding vectors
    if(not use_preprocessed and train_wv): # train new vectors using corpus
        # clean the corpus text
        corpus_path = cfg.DirConfig.CORPUS_PATH_DICT[corpus_mode]
        clean_corpus_path = cfg.DirConfig.CLEAN_CORPUS_PATH_DICT[corpus_mode]
        
        text_cleaner = du.TextCleaner(corpus_path)
        # load stop word list into text cleaner if needed
        if s_w_rmvl:
            stop_set = vocab.get_stop_word_set()
        
        text_cleaner.clean_chn_corpus_2file(clean_corpus_path, stop_set)
        
        # copy cleaned text to word2vec directory
        clean_corpus_filename = cfg.DirConfig.CLEAN_CORPUS_FILENAME_DICT[corpus_mode]
        dst_corpus_path = cfg.DirConfig.W2V_DIR + clean_corpus_filename
        copyfile(clean_corpus_path, dst_corpus_path)
        
        vocab.build_vocab_database(qa_data_mode, clean_corpus_filename)
    else:
        vocab.load_wv_from_db()
    
    '''
     initialize data stream for model
     data stream initiates unknown-word mapping
     so beware of deadlock in database operation
    '''
    """  
    # clean question-answer pair text
    if(train_set != 'NAH' and not use_preprocessed):
        text_cleaner = du.TextCleaner(qa_data_path_t)
        text_cleaner.clean_chn_text_2file(clean_qa_data_path_t, stop_set)
    if(eval_set != 'NAH' and not use_preprocessed):
        text_cleaner = du.TextCleaner(qa_data_path_e)
        text_cleaner.clean_chn_text_2file(clean_qa_data_path_e, stop_set)
    
    '''training and evaluating model'''
    if train_set != 'NAH': # start training
        data_stream = du.SentenceDataStream(clean_qa_data_path_t, vocab, (qa_data_mode, 't'))
        
    if  eval_set != 'NAH': # start evaluating
        data_stream = du.SentenceDataStream(clean_qa_data_path_e, vocab, (qa_data_mode, 'e'))
    
    """
    '''
    sentence = ["狐狸", "吃", "apple"]    
    idx_seq = vocab.to_idx_sequence(sentence)
    print(idx_seq)
    ret = vocab.get_um()
    print(ret[0])
    print(ret[1])
    sentence = ["西瓜", "sweet"]
    idx_seq = vocab.to_idx_sequence(sentence)
    print(idx_seq)
    ret = vocab.get_um()
    print(ret[0])
    print(ret[1])
    '''
    
    write_log("Finished.\n")
    
def generate_paths(qa_data_mode, ling_unit, s_w_rmvl):
    '''
     generate a tuple of paths needed
     including corpus path, cleaned corpus path, vector database path,
     training data path, evaluation data path
    pass
    '''
    pass
    

