# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:53:19 2017

@author: merry jane

1、训练lstm 生成文本模型

"""

from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras
import numpy as np
#from keras.datasets import imdb
from keras import backend as K
import re
import jieba
import pickle
import data_helper




def build_lstm_model(batch_size=256):
    '''
    arg:
        char_indices:输入文本的词语索引字典
        maxlen:每一行文本按词语最大长度
        
    
    模型如图所示，利用keras搭建，tensorflow作为backend
    ________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_24 (LSTM)               (None, 256)               13986816  
    _________________________________________________________________
    dense_14 (Dense)             (None, 13402)             3444314   
    _________________________________________________________________
    activation_14 (Activation)   (None, 13402)             0         
    =================================================================
    Total params: 17,431,130
    Trainable params: 17,431,130
    Non-trainable params: 0
    _________________________________________________________________
    '''
    
    
    
    
   
    batch_size=batch_size
    #print ('build model...')
    model=Sequential()
    model.add(LSTM(256,input_shape=(maxlen,len(char_indices)),recurrent_dropout=0.1,dropout=0.1))
    model.add(Dense(len(char_indices)))
    model.add(Activation('softmax'))
    
    optimizer=keras.optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer)
    
    #print (model.summary())
    
    return model


def train_model(X,y,batch_size=256,epochs=10):
    '''
    小批量训练fit_generator
    '''
    model.fit_generator(data_generator(X,y,batch_size),steps_per_epoch=X.shape[0]//batch_size,epochs=epochs)




def data_generator(X,y,batch_size):
    '''
    数据较小批量生成
    '''
    if batch_size<1:
        batch_size=256
    number_of_batchs=X.shape[0]/batch_size
    counter=0
    shuffle_index=np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    
    #reset generator
    while 1:
        index_batch=shuffle_index[batch_size*counter:batch_size*(counter+1)]
        X_batch=(X[index_batch,:,:]).astype('float32')
        y_batch=(y[index_batch,:]).astype('float32')
        counter+=1
        yield(np.array(X_batch),y_batch)
        
        if (counter<number_of_batchs):
            np.random.shuffle(shuffle_index)
            counter=0
            
            
def gen_matrix_for_sentence():
    #对构造句子 对其矩阵化
    print ('vectorization...')
    X=np.zeros((len(sentences),maxlen,len(char_indices)),dtype=np.bool)
    y=np.zeros((len(sentences),len(char_indices)),dtype=np.bool)
    
    for i,sentence in enumerate(sentences[:]):
        if (i%30000==0):
            print (i)
            for t in range(maxlen):
                char_index=sentence[t]
                X[i,t,char_index]=1
    
        y[i,next_chars[i]]=1
    
    return X,y



        
        
if __name__=='__main__':
    
    #1、读文章数据
    #############alltext只要是文本即可 也可以留下文本标点###################
    f=open('./fxhh_article.csv','r',encoding='utf-8')
    line=f.readline()
    alltext=''
    i=0
    while True:
        line=f.readline()
        if not line:break
        i+=1
        
        #剔除标点符号
        alltext+=line.split('\x01')[2]+"".join([x.strip() for x in re.findall('(.*?)[|,|;|?|.|!|:|，|。|、|~|；|！|？|"|“|”|】|【|《|》|<|>|&|%|$|(|)|（|）|^|#|@]', line.split('\x01')[3] + ",")])
        
        #防止内存溢出 在gen_matrix_for_sentence
        if i>2000:break
    
    
    ###############################################################
    
    #2、生成词与索引，索引与词的映射字典
    data_helper.preprocess_and_save_data(alltext, data_helper.create_lookup_tables) #生成并保存
    int_text, char_indices, indices_char = data_helper.load_preprocess() #加载 int_text即是包含所有词语索引号的数组
    
    #3、构造句子序列 int_text, vocab_to_int, int_to_vocab
    maxlen=10 #每个句子包含词语的最大个数
    step=3 #跳跃生成
    sentences=[]
    next_chars=[]
    for i in range(0,len(int_text)-maxlen,step):
        sentences.append(int_text[i:i+maxlen])
        next_chars.append(int_text[i+maxlen])
    
    print ('nb sequences:',len(sentences))

    #4、生成句子转矩阵  
    X,y=gen_matrix_for_sentence()
    
    #5、bulid模型   默认batch_size=256 小批量处理的句子树
    model=build_lstm_model(batch_size=256)
    
    #6、训练模型 epochs为训练轮次
    train_model(X,y,batch_size=256,epochs=1)
    
    
    #7、保存模型
    model.save('./lstm_model_gen_word_epoch_10.h5')
    
    

    


