# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:15:19 2017

@author: merry jane

数据导入工具 helper
"""
import jieba
import pickle
import re 

def create_lookup_tables(input_data):
    
    vocab = set(input_data)
    #print ('vocab,',vocab)
    
    # 文字到数字的映射
    vocab_to_int = {word: idx for idx, word in enumerate(vocab)}
    
    # 数字到文字的映射
    int_to_vocab = dict(enumerate(vocab))
    
    return vocab_to_int, int_to_vocab

def preprocess_and_save_data(text,  create_lookup_tables):
    '''
    arg:
        text 所有输入文本
        create_lookup_tables 用于生成字典索引映射 函数
    
    '''
    text=jieba.lcut(text)
    p = re.compile(r'\d+')
    text=[word for word in   filter(lambda x:len(re.findall(p,x))==0 ,text)]  

    #print (text)
    
    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    
    pickle.dump((int_text, vocab_to_int, int_to_vocab), open('./preprocess.p', 'wb'))
    
    
    
def load_preprocess():
    return pickle.load(open('preprocess.p', mode='rb'))
