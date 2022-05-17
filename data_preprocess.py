import pandas as pd
import numpy as np
import re
import os
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

### 1 文本去噪
#去除一些无用的信息；特殊的符号
def clean_noise(text):
    text = text.replace(u'\xa0', u' ')      # 去除 \xa0     不间断空白符 
    text = text.replace(u'\u3000', u' ')    # 去除 \u3000   全角的空白符
    
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", "", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"\[\S+\]", "", text)     # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)     # 去除话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)      # 去除网址
    
    EMAIL_REGEX = re.compile(r"[-a-z0-9_.]+@(?:[-a-z0-9]+\.)+[a-z]{2,6}", re.IGNORECASE)
    text = re.sub(EMAIL_REGEX, "", text)    # 去除邮件 
    
    text = text.replace("转发微博", "")      # 去除无意义的词语
    text = text.replace("网页链接", "")
    text = re.sub(r"\s+", " ", text)        # 合并正文中过多的空格

    text = re.sub(r"\d{2,4}年|\d{1,2}月|\d{1,2}日|\d{1,2}时|\d{1,2}分| \d{1,2}点", "", text) # 去除 日期 时间
    text = re.sub(r"\d", "", text)
    text = re.sub('[a-zA-Z]',"", text)
    
    return text.strip()

### 2/3 分词 过滤停用词
# 读取停用词列表
def get_stopword_list(file):
    with open(file, 'r', encoding='utf-8') as f:    # 
        stopword_list = [word.strip('\n') for word in f.readlines()]
    return stopword_list

# 分词 然后清除停用词语
def clean_stopword(str, stopword_list):
    result = []
    word_list = jieba.cut(str)   # 分词后返回一个列表  jieba.cut(）   返回的是一个迭代器
    for w in word_list:
        if w not in stopword_list:
            result.append(w)
   
    return result


### 7.特征词选择
# 统一处理和学习，这个就比较麻烦了

def word2vec(lsls):
    ls_str=[]
    for s in lsls:
        strs = ' '.join(s) 
        ls_str.append(strs)
    
    # TF-IDF
    transfer = TfidfVectorizer() #实例化一个转换器类
    data_new = transfer.fit_transform(ls_str) #调用fit_transform()
    #构建成一个二维表：
    df=pd.DataFrame(data_new.toarray(), columns=transfer.get_feature_names_out())
    
    withWeight = transfer.vocabulary_
    return df, withWeight

def main():
    file = '完整数据_暴雨_交通.csv'
    path_file = os.path.join("Data",file)

    data = pd.read_csv(path_file)
    df = data.sample(n=500, replace=False, random_state=1)
    df.reset_index(drop=True, inplace=True)
    tweets = df.loc[:, ['微博正文']]
    tweets.shape

    tweets['clean_word'] = tweets['微博正文'].apply(clean_noise)
    stopword = get_stopword_list('stopwords/hit_stopwords.txt')
    tweets['clean_stopwords'] = tweets['clean_word'].apply(clean_stopword
                                                          ,stopword_list=stopword)  
    
    lsls = tweets['clean_stopwords']
    word_v, withWeight = word2vec(lsls)

    topK = 500  # 排名前几的
    downK = int(len(withWeight)/4) #排名靠后的删掉
    L = sorted(withWeight.items(), key=lambda item:item[1], reverse=False)

    tags_del = list(map(lambda x: x[0], L[0:downK]))
    tags_del

    df2 = word_v.drop(columns=tags_del, axis=1, inplace=False)
    
    tweets.to_excel(os.path.join("Data", file[:-4]+"_pre_tweets.xlsx"))
    df2.to_excel(os.path.join("Data",file[:-4]+"_pre_vec.xlsx"))


if __name__ == '__main__':
    main()