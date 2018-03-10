#! -*- coding:utf-8 -*-
'''

Data processing for model

@author: LouisZBravo

'''
import numpy as np
import config as cfg
import sqlite3
import re

import sys
import os
ext_tool_dir = cfg.DirConfig.EXT_TOOL_DIR
sys.path.append(os.path.abspath(ext_tool_dir))
# ignore error msg here as long as jieba package in the right place
import jieba

class TextCleaner(object):
    '''
     text cleaner instance
     clean raw text for both vocab and sentence match model
     tokenization -> stop words removal
    '''
    def __init__(self, text_path):
        self.text_path = text_path
        self.ling_unit = cfg.PreProcessConfig.LING_UNIT
        assert self.ling_unit in ["WORD", "CHAR"]
        self.punc_rmvl = cfg.PreProcessConfig.PUNCTUALATION_REMOVAL
        self.s_w_rmvl = cfg.PreProcessConfig.STOP_WORD_REMOVAL
        self.num_rmvl = cfg.PreProcessConfig.NUMBER_REMOVAL
        #to_lemmatize?
        #to tokenize
        
    
    def next_sentence(self):
        pass
    
    def unk_mapping(self, vocab):
        pass
    
    def clean_doc(self, doc_path, use_cached):
        '''
         clean the document provided
         a full process includes segmentation, tokenizing, stop-word removal, number and proper noun processing
         for data streaming use
         return a piece of cleaned text
        '''
        # for w2v training & model
        # save both to cached & w2v dir
        
        '''
         for english, we need to lowercase all letters
        '''
        
        return
    
    def clean_chn_text_2file(self, save_path, stop_set):
        # s_w_rmvl = cfg.PreProcessConfig.STOP_WORD_REMOVAL
        f = open(self.text_path, 'rb') # use 'rb' mode for windows decode problem
        f_w = open(save_path, 'wb')
        # start to clean each line
        for l in f:
            line = l.decode("utf-8")
            # remove punctualation
            if self.punc_rmvl:
                # replace punctuation with spacing
                # to make separate items remain separated
                line = re.sub("[][！？。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@\\\^_`{|}~\s\t]+", " ", line)
            # segmentation based on word or char
            line_seg = []
            if self.ling_unit == "WORD": # seg the text by word
                for w in jieba.cut(line):
                    if(w != " " and w not in stop_set):
                        
                        # make all number zero
                        if self.num_rmvl:
                            word = self._remove_digit(w)
                        else:
                            word = w
                        
                        line_seg.append(word)
            else: # seg the text by char
                for ch in line:
                    if(ch != " " and ch not in stop_set):
                        
                        # make all number zero
                        if self.num_rmvl:
                            char_ = self._remove_digit(ch)
                        else:
                            char_ = ch
                            
                        line_seg.append(char_)
            print(line_seg)
            if len(line_seg) == 0:
                continue
            f_w.write(' '.join(line_seg).encode("utf-8"))
            f_w.write(' '.encode("utf-8"))
            
            
        f.close()
        f_w.close()
        
    def _remove_digit(self, s):
        flag_all_digit = True
        for ch in s:
            if(ch < '0' or ch > '9'):
                flag_all_digit = False
                break
        if flag_all_digit:
            return '0'
        else:
            return s

class Vocab(object):
    '''
     Vocabulary instance for model
     
     Important attributes:
     word_matrix: the word matrix composed by concatenating words' distributional representation vectors
     word2id: map from word to id in the word matrix
     id2word: map from id to word in the word matrix
     word_dim: dimensionality of a word vector, number of columns in word matrix
     vocab_size: number of known words, number of rows in word matrix
     stop_set: a set of stop words
     __unk_mapping: map of OOV words to known words
     
     Will expand to cover all vocabulary appearing in text
    '''    
    def __init__(self, wv_path):
        self.word2idx = {}
        self.idx2word = {}
        self.wdim = 0
        self.vocab_size = 0
        self.wv_path = wv_path
        self.word_matrix = None
        self.stop_set = self.load_stop_word_set()
        self.__unk_mapping = {}
        
    
    def build_vocab_database(self, qa_data_mode, clean_corpus_filename):
        '''
         train word vectors on corpus and save them into database
         WORD_VECTORS:
         WORD TEXT | DIM0 REAL | DIM1 REAL | ... | DIMn REAL
         
         VECTOR_DIMENSIONALITY:
         WDIM
         
         VOCABULARY_SIZE(invisible):
         VSIZE
         
         UNKNOWN_WORD_MAPPING:
         WORD | SIMILAR_WORD
        '''
        
        wdim = cfg.ModelConfig.WORD_DIM_DICT[qa_data_mode]
        cbow = cfg.PreProcessConfig.W2V_ALGORITHM_CBOW
        win_size = cfg.PreProcessConfig.W2V_WIN_SIZE
        neg_samp = cfg.PreProcessConfig.W2V_NEG_SAMP
        hs = cfg.PreProcessConfig.W2V_HIER_SFTMX
        iter_ = cfg.PreProcessConfig.W2V_ITER
        
        #execute word2vec program
        print('Please launch the word2vec program')
        #write_log('Please launch the word2vec program')
        print('using parameters:')
        #write_log('using parameters:')
        print('-train %s -output vectors.bin -cbow %d -size %d -window %d -negative %d -hs %d -sample 1e-4 -threads 20 -binary 0 -iter %d\n' % (
            clean_corpus_filename, cbow, wdim, win_size, neg_samp, hs, iter_))
        #write_log('-train %s -output vectors.bin -cbow %d -size %d -window %d -negative %d -hs %d -sample 1e-4 -threads 20 -binary 0 -iter %d\n' % (clean_corpus_filename, cbow, wdim, win_size, neg_samp, hs, iter_))
        print('Press enter after word2vec program successfully generates vectors binary file "vecters.bin".\n')
        input()
        
        print ('Start reading vector data from word2vec program...')
        #write_log('Start reading vector data from word2vec program...')
        
        w2v_res_path = cfg.DirConfig.W2V_RES_DIR
        vec_bin_file = open(w2v_res_path, 'rb')
        line = vec_bin_file.readline()
        vsize = int(line.split()[0])
        wdim_read = int(line.split()[1])
        assert wdim == wdim_read
        
        line = vec_bin_file.readline()
        widx = 0
        vector_list = []
        while(line != b''):
            word = line.split()[0]
            vector_b = line.split()[1:]
            vector = [float(i) for i in vector_b]
            vector_list.append(vector)
            self.word2idx[word] = widx
            self.idx2word[widx] = word
            widx += 1
            line = vec_bin_file.readline()
            
        self.vocab_size = vsize # widx?
        self.wdim = wdim_read
        
        vec_bin_file.close()
        
        # initialize a empty word matrix 
        # zero vector is the last row
        self.word_matrix = np.zeros((self.vocab_size + 1, self.wdim), 
                                    dtype = np.float32)
        for cur_idx in range(self.vocab_size):
            self.word_matrix[cur_idx] = vector_list[cur_idx] # each row is a word vector
            
        print('Word vectors successfully loaded. Now saving to database...')
        
        conn = sqlite3.connect(self.wv_path)
        c = conn.cursor()
        
        # create table for word vectors
        dim_col_strs = "" # string of column names to be filled into command
        for i in range(wdim_read):
            dim_col_strs = dim_col_strs + ", DIM%d REAL" % i #, dim0 real, dim1 real...
        # CREATE TABLE word_vecs (word text, dim0 real, ..., dimn real)
        crt_tbl_command = "CREATE TABLE WORD_VECTORS (WORD TEXT" + dim_col_strs + ")"
        c.execute(crt_tbl_command)
        
        # insert vector values into database
        q_marks = ""
        for i in range(self.wdim):
            q_marks = q_marks = q_marks + ", ?"
            
        ist_val_command = "INSERT INTO WORD_VECTORS VALUES (?" + q_marks + ")"
        wvec2db_lst = []
        for w_idx in range(self.vocab_size):
            wvec2db = [self.idx2word[w_idx]] # ['word']
            wvec2db.extend(vector_list[w_idx]) # ['word', dim1, dim2...]
            wvec2db_lst.append(tuple(wvec2db))
        
        c.executemany(ist_val_command, wvec2db_lst)
        conn.commit()
        
        # record vector dimensionality & vocab size
        crt_tbl_command = "CREATE TABLE VECTOR_DIMENSIONALITY (WDIM INTEGER)"
        c.execute(crt_tbl_command)
        crt_tbl_command = "CREATE TABLE VOCABULARY_SIZE (VSIZE INTEGER)"
        c.execute(crt_tbl_command)
        ist_val_command = "INSERT INTO VECTOR_DIMENSIONALITY VALUES (?)"
        c.execute(ist_val_command, (wdim_read,))
        ist_val_command = "INSERT INTO VOCABULARY_SIZE VALUES (?)"
        c.execute(ist_val_command, (vsize,))
        
        
        # create unknown word mapping
        crt_tbl_command = "CREATE TABLE UNKNOWN_WORD_MAPPING (WORD TEXT, SIMILAR_WORD TEXT)"
        c.execute(crt_tbl_command)
        
        conn.commit()
        
        conn.close()
        
    
    def load_wv_from_db(self):
        '''
         Load pre-trained word vectors from database
         WORD_VECTORS:
         WORD TEXT | DIM0 REAL | DIM1 REAL | ... | DIMn REAL
         
         VECTOR_DIMENSIONALITY:
         WDIM
         
         VOCABULARY_SIZE(invisible):
         VSIZE
         
         UNKNOWN_WORD_MAPPING:
         WORD | SIMILAR_WORD
        '''
        
        wv_path = self.wv_path
        conn = sqlite3.connect(wv_path)
        c = conn.cursor()
        
        # get word dimensionality
        c.execute("SELECT * FROM VECTOR_DIMENSIONALITY")
        (wdim, ) = c.fetchone()
        assert c.fetchone() == None
        self.wdim = wdim
        
        # get vocab size
        c.execute("SELECT * FROM VOCABULARY_SIZE")
        (vsize, ) = c.fetchone()
        assert c.fetchone() == None
        self.vocab_size = vsize
        
        # get word, word vectors and count their indices
        widx = 0
        vector_list = []
        for row in c.execute("SELECT * FROM word_vectors"):
            word = row[0] # row[0].decode('utf-8')
            vector = row[1:]
            vector_list.append(vector)
            self.word2idx[word] = widx
            self.idx2word[widx] = word
            widx += 1
            
        # get oov word mapping
        for oov_word, sim_word in c.execute("SELECT * FROM UNKNOWN_WORD_MAPPING"):
            self.__unk_mapping[oov_word] = sim_word
            
        conn.close()
        
        self.vocab_size = widx
        
        # initialize a empty word matrix 
        # zero vector is the last row
        self.word_matrix = np.zeros((self.vocab_size + 1, self.wdim), 
                                    dtype = np.float32)
        for cur_idx in range(self.vocab_size):
            self.word_matrix[cur_idx] = vector_list[cur_idx] # each row is a word vector
            
            if cur_idx % 20 == 0:
                print("%d: %s" % (cur_idx, self.idx2word[cur_idx].decode("utf-8")))
                print(self.word_matrix[cur_idx])
        
        
    def has_word(self, word):
        ret = word in self.word2idx
        return ret
    
    def get_um(self):
        # for testing
        conn = sqlite3.connect(self.wv_path)
        c = conn.cursor()
        cmd = "SELECT * FROM UNKNOWN_WORD_MAPPING"
        c.execute(cmd)
        db_ = c.fetchall() # from db
        ret = (db_, self.__unk_mapping)
        return ret
    
    def unk_map(self, oov_word):
        '''
         need to be re-wrote
         unknown words are generated during text cleaning of sentence pair file
        '''
        if oov_word in self.__unk_mapping:
            sim_word = self.__unk_mapping[oov_word]
        else: # not recorded
            # randomly map this word to a similar word
            sim_idx = np.random.randint(0, self.vocab_size)
            sim_word = self.idx2word[sim_idx]
            # store oov word mapping
            self.__unk_mapping[oov_word] = sim_word
            # save mapping res in db
            conn = sqlite3.connect(self.wv_path)
            c = conn.cursor()
            ist_val_command = "INSERT INTO UNKNOWN_WORD_MAPPING VALUES (?, ?)"
            c.execute(ist_val_command, (oov_word, sim_word))
            conn.commit()
            conn.close()
        return sim_word
    
    def load_stop_word_set(self):
        stop_set = set([])
        s_w_dir = cfg.DirConfig.STOP_WORD_DIR
        try:
            f = open(s_w_dir, 'rb')
        except FileNotFoundError:
            write_log("Stop word file not found while initializing vocab before training word vectors.\n")
        else: 
            # every line is a stop word
            for l in f:
                line = l.decode("utf-8")
                s_w = line.strip('\n')[0]
                stop_set.add(s_w)
            f.close()
        return stop_set
    
    def get_stop_word_set(self):
        return self.stop_set
            
    def to_idx_sequence(self, sentence):
        '''
         transfer a sentence(list of word) to a sequence of indices
         linguistic unit in sentence separated by spacing
         specific condition handling strategy as follows:
         out-of-vocab word:
             seen as a random invariant idx
             db <- mem <- new found OOV
         vocab cleaning:
             ...
         stop word:
             3 approaches:
             using their original vector representations
             regarding them as OOV words
             totally ignore them
        '''
        idx_sequence = []
        for word in sentence:
            if self.has_word(word): # in-vocab word
                pass # stop process & cleaning
                idx = self.word2idx[word]
            else: # OOV word
                sim_word = self.unk_map(word)
                idx = self.word2idx[sim_word]
                
                
            idx_sequence.append(idx)
        
        return idx_sequence

class SentenceDataStream(object):
    '''
     data stream of sentence pairs, made into batches
     words in sentence are stored as indices by the Vocab instance
     sentence pairs are sorted by sentence length
     important attributes:
         instance: list of all sentence pairs, sorted as above
         -----remember to make space for preprocess customization set by config parameters-----
         batches: iterator? of data to be sent into the model
         batch_span: list of tuples pointing positions of each batch in instance list
    '''
    def __init__(self, clean_qa_file_path, vocab, mode):
        '''
         read question answer pair file and save qa pairs into memory
         mode: a tuple of dataset name and train/evaluation mode, e.g. ('HITNLP', 't')
        '''
        
        qa_data_mode = mode[0]
        t_e_mode = mode[1]
        assert qa_data_mode in cfg.ModelConfig.SUPPORTED_DATASET
        assert t_e_mode in ('t', 'e')
        
        to_sort = cfg.ModelConfig.SORT_INSTANCE
        
        self.instances = []
        self.t_e_mode = t_e_mode
        self.batch_span = []
        self.q_batch_lst = []
        self.a_batch_lst = []
        self.label_batch_lst = []
        
        f = (open(clean_qa_file_path, 'rb'))
        
        if qa_data_mode == 'HITNLP':
            for line in f:
                line_item = line.split('\t')
                q = line_item[0]
                a = line_item[1]
                if t_e_mode == 't':
                    label = int(line_item[2])
                q_idx_seq = vocab.to_idx_sequence(q.split(' '))
                a_idx_seq = vocab.to_idx_sequence(q.split(' '))
                if t_e_mode == 't':
                    self.instance.append((q_idx_seq, a_idx_seq, label))
                else:
                    self.instance.append((q_idx_seq, a_idx_seq)) # in evaluation mode results are generated by model and saved later
        else:
            pass # other dataset in different format
        
        # sort instances
        if to_sort:
            self.instances = sorted(self.instances, key = lambda instances: (len(instances[0]), len(instances[1])))
        """ does variable length affects performance of convolution filter? """
        # make batches
        
    def make_patch_span(self, batch_size, instance_size):
        batch_span = []
        batch_num = int(np.ceil(float(instance_size) / float(batch_size)))
        for i in range(batch_num):
            batch_span.append((i * batch_size, min(instance_size, (i+1)*batch_size)))
        return batch_span





def clean_chn_words(word):
    pass

def write_log(log_str):
    log_dir = cfg.DirConfig.LOG_DIR
    log_file_path = cfg.DirConfig.LOG_FILE
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    f = open(log_file_path, 'a')
    f.write(log_str)
    f.close()