# 导入包
# 导入数据

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import pkuseg
import jieba
import os 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


def clear_noise(text):
    """
    text 待去噪的原始文本
    """
    text = text.replace(u'\xa0', u' ')      # 去除 \xa0     不间断空白符 
    text = text.replace(u'\u3000', u' ')    # 去除 \u3000   全角的空白符
    
    # 匹配这些中文标点符号 。 ？ ！ ， 、 ； ： “ ” ‘ ' （ ） 《 》 〈 〉 【 】 『 』 「 」 ﹃ ﹄ 〔 〕 … — ～ ﹏ ￥
    reg1 = r"@\S*?"
    reg2 = r'[\s|\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]'
    text = re.sub(r"(回复)?(//)?\s*", "", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(reg1+reg2, "", text)
    text = re.sub(r"\[\S+\]", "", text)     # 去除表情符号
    # text = re.sub(r"#\S+\s*#", "", text)    # 去除话题内容 “#content#”
    # text = re.sub(r"【\S+\s*】", "", text)  # 分类标签“【content】”

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
    text = re.sub(r'[a-zA-Z]',"", text)
    
    return text.strip()




# jieba分词

def jieba_segment(str, use_paddle=True):
    """
    jieba分词:
    str         : 待分词的文本
    use_paddle  : 是否使用paddle模型
    """
    word_list = jieba.cut(str, use_paddle=use_paddle)   # 分词后返回一个列表  jieba.cut(）   返回的是一个迭代器
    res = list(word_list)
   
    return res


# jieba 词性标注（part-of-speech tagging）

def jieba_postag(str, use_paddle=True):
    """
    jieba分词:
    str         : 待分词的文本
    use_paddle  : 是否使用paddle模型
    """
    import jieba.posseg as pseg
    word_tag_ls = pseg.lcut(str, use_paddle=use_paddle)
    
    res = [] # 将词语与词性从pair对象转换为 元组，方便索引
    for w_t in word_tag_ls:
        w, t = w_t
        res.append((w, t))
    return res

# pkuseg 分词 

def pkuseg_segment(str, m_name="web", u_dict="default"):
    """
    pkuseg分词:
    str     :待分词的文本
    m_name  : 选择使用的预训练模型
    u_dict  : 用户自定义的分词设置用户词典。
    """
    # m_name= "news"
    # m_name= "mixed"
    # p_tag= True # 是否包含词性
    seg = pkuseg.pkuseg(model_name=m_name, user_dict=u_dict)           # 以默认配置加载模型    
    res = seg.cut(str)  # 进行分词
   
    return res

# pkuseg 词性标注（part-of-speech tagging）
# pkuseg 分词 

def pkuseg_postag(str, m_name="web", u_dict="default", postag=True):
    """
    pkuseg分词:
    str     :待分词的文本
    m_name  : 选择使用的预训练模型
    u_dict  : 用户自定义的分词设置用户词典。
    postag   : 是否进行词性标准（是的话，需要在自定义的词典中，也要添加相应的词性 tab键隔开在一行
    """
    # m_name= "news"
    # m_name= "mixed"
    # postag= True # 是否包含词性
    seg = pkuseg.pkuseg(model_name=m_name, user_dict=u_dict, postag=postag)           # 以默认配置加载模型    
    res = seg.cut(str)  # 进行分词
   
    return seg.cut(str)  # 进行分词



def clear_stopword(word_ls, stopword_file, postag=False, stop_postag=[], user_file="../05stopwords/user_stopwords.txt"):
    """
    word_ls         :待去停用词的词汇列表【word 或者是 tuple(word, postag) 组成的list
    stopword_file   :选择使用的停用词库
    user_file       :用户自定义的停用词库
    postag          :是否有词性标注
    stop_postag     :指定删除的词性
    """
    with open(stopword_file, 'r', encoding='utf-8') as f1, open(user_file, 'r' , encoding='utf-8') as f2:    # 
        
        stopword_ls = [word.strip('\n') for word in f1.readlines()] # 默认词库
        user_ls = [word.strip('\n') for word in f2.readlines()]     # 自定义词库

        stopword_ls.extend(user_ls)
        
        res = []
        if postag: # 有词性标注
            for word_tag in word_ls:
                w, t = word_tag
                if w not in stopword_ls and len(w) > 1 and t not in stop_postag: # 仅保留2个字符及以上的词
                    res.append((w, t))
        else:
            for w in word_ls:
                if w not in stopword_ls and len(w) > 1: # 仅保留2个字符及以上的词
                    res.append(w)
    
    return res



# 向量化
def wordVectorizer(word_ls_column, v_type, postag=True):
    ls_str=[]
    if postag:
        for word_tag_ls in word_ls_column:
            if word_tag_ls == []:
                word_tag_ls = [('','')]
            word_ls = np.array(word_tag_ls)[:, 0].tolist()
            tag_ls = np.array(word_tag_ls)[:, 1].tolist()
            
            strs = ' '.join(word_ls)
            ls_str.append(strs)
    else:
        for word_ls in word_ls_column:
            strs = ' '.join(word_ls) 
            ls_str.append(strs)
    
    if v_type.upper() == "TFIDF":
    # TF-IDF(term frequency—inverse document frequency)
        transfer = TfidfVectorizer() #实例化一个转换器类
    else:
        transfer = CountVectorizer()
    data_new = transfer.fit_transform(ls_str) #调用fit_transform()
    #构建成一个二维表：
    df=pd.DataFrame(data_new.toarray(), columns=transfer.get_feature_names_out())
    
    withWeight = transfer.vocabulary_
    return df, withWeight

if __name__ == '__main__':

    print("============ 改变路径: weibo-analysis")
    print(os.getcwd())
    os.chdir("./03Data_Process")
    print(os.getcwd())
    print("============ 改变路径: 03Data_Process")

    data = pd.read_csv("../02Data/完整数据_暴雨_交通.csv")
    df = data.sample(n=500, replace=False, random_state=1) # 指定随机种子，保证都是同一个数据组合
    df.reset_index(drop=True, inplace=True)


    # tweets = df.loc[:99, ['微博正文']].astype("string") #! 取100条数据就好了
    tweets = df.loc[:, ['微博正文']].astype("string")

    tweets["去噪"] = tweets["微博正文"].apply(clear_noise).astype("string")

    tweets["jieba"] = tweets["去噪"].apply(jieba_segment)
    # tweets["pkuseg"] = tweets["去噪"].apply(pkuseg_segment)
    
    tweets["jieba_postag"] = tweets["去噪"].apply(jieba_postag)
    # tweets["pkuseg_postag"] = tweets["去噪"].apply(pkuseg_postag)


    stopword_file = "../05stopwords/hit_stopwords.txt"
    user_file = "../05stopwords/user_stopwords.txt"
    
    postag = False
    stop_postag = []
    tweets["jieba_stop"] = tweets["jieba"].apply(clear_stopword, args=[stopword_file, postag, stop_postag, user_file])
    # tweets["pkuseg_stop"] = tweets["pkuseg"].apply(clear_stopword, args=[stopword_file, postag, stop_postag, user_file])
    
    postag = True
    stop_postag= ['c','d','df','f','m','mq','ns','nt','nz','nr','o','p','q','r','s','t','nrfg','nrt']

    tweets["jieba_postag_stop"] = tweets["jieba_postag"].apply(clear_stopword, args=[stopword_file, postag, stop_postag, user_file])
    # tweets["pkuseg_postag_stop"] = tweets["pkuseg_postag"].apply(clear_stopword, args=[stopword_file, postag, stop_postag, user_file])


    # tweets.to_excel("../02Data/jieba_pkuseg2.xlsx")
    tweets_file = "../02Data/02完整数据_暴雨_交通_pre_tweets_TFIDF.xlsx"
    tweets.to_excel(tweets_file)

    # 向量化
    word_ls_column = tweets['jieba_postag_stop']
    v_type = 'TFIDF'
    postag = True
    word_v, withWeight = wordVectorizer(word_ls_column, v_type, postag)

    downK = int(len(withWeight)/4) # 排名靠后1/4的删掉
    L = sorted(withWeight.items(), key=lambda item:item[1], reverse=False)

    tags_del = list(map(lambda x: x[0], L[0:downK]))
    print(len(tags_del))
    

    res = word_v.drop(columns=tags_del, axis=1, inplace=False)


    print(word_v.shape)
    print(res.shape)

    res_file = "../02Data/02完整数据_暴雨_交通_pre_vec_TFIDF.csv"
    res.to_csv(res_file, index=None)
    
