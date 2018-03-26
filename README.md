# Convolutional-Neural-Network-for-QA-Pair-Selection
Implementation of A. Severyn et al, "Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks", 2015 

The system works in a predefined directory structure:
Basically, its root should include 3 folders: 
src\
data\
ext_tool\

config.py, data_util.py, model.py, model_control.py, model_interface are supposed tp be put in "src\".
External tools including jieba(module in folder), evaluation module(evaluation.py, by yafuilee), word2vec(by T. Mikolov) are supposed to be put in "ext_tool\".

in "data\", there are 2 folders and 1 weight_file:
data\raw\
data\cached\

Question pair files and stop word file(named "stop_words") are supposed to be put in "data\raw\". (for copyright reason, I will only release them if granted)

During execution, several files and folders will be automatically generated:
log\
log\log.txt
data\cached\*.db
data\cached\*cleaned_corpus*
data\my_model_weights.h5
data\*.score
