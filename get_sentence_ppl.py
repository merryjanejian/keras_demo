# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:45:10 2017

@author: merry jane

调用训练好的lstm模型 预测句子中每个词语的概率和ppl
2、输入句子，返回句子词语 相对出现概率 和 总的通顺性 ppl(概率的几何平均值)
"""
from keras.models import load_model
import jieba
import re
import data_helper


def get_words_prob(sentence):
    '''
    预测句子每个词条件概率，该句子的通顺性值=1/词语概率乘积
    '''
    
    sentence="".join([x.strip() for x in re.findall('(.*?)[|,|;|?|.|!|:|，|。|、|~|；|！|？|"|“|”|】|【|《|》|<|>|&|%|$|(|)|（|）|^|#|@]', sentence+',')])
    sen_words=jieba.lcut(sentence)
    p = re.compile(r'\d+')
    sen_words=[word for word in   filter(lambda x:len(re.findall(p,x))==0 ,sen_words)] 
    word_prob={}
    word_prob_list=[]
    i=0
    ok_w=[]
    for w in sen_words:
        if i >=len(sen_words):
            break
            
        next_w=sen_words[i]
        
        input_words=["的"]*(maxlen-i)+ok_w[-maxlen:]
        x=np.zeros((1,maxlen,len(char_indices)))
    
        for t,char in enumerate(input_words):
            try:
                x[0,t,char_indices[char]]=1.
            except:
                continue

        i+=1
   
   

        preds=model.predict(x,verbose=0)[0]
        try:
            word_prob[next_w]=preds[char_indices[next_w]]/np.mean(preds)
        except:
            word_prob[next_w]=0.0
        word_prob_list.append( word_prob[next_w])
        
        ok_w.append(w)
    
    
    return word_prob,(np.array(word_prob_list).prod())**(1/len(word_prob_list))




if __name__=='__main__':
    
    model = load_model('./lstm_model_gen_word_epoch_10.h5')
    _, char_indices, indices_char = data_helper.load_preprocess()
    get_words_prob('讲究简单，朴素无华，细腻光滑工艺外壳，光泽温和，质感醉厚，')
    
    
    '''
    返回结果：
    ({'光泽': 4.7096591,
  '光滑': 5.9352827,
  '外壳': 2.7460401,
  '工艺': 21.515692,
  '朴素无华': 0.16896033,
  '温和': 3.5096765,
  '简单': 14.434443,
  '细腻': 26.224731,
  '讲究': 0.081697658,
  '质感': 19.405516,
  '醉厚': 0.0076282052},
   2.1476565351778878)
    '''
