# weibo-tweets-analysis

## 文本处理流程

> 代码：
>
> [test.ipynb](./test.ipynb)  【 全流程的学习和测试代码
>
> 参考文献：
>
> - 基于微博数据的高速公路交通事件研究_包丹
> - 基于社交网络数据的交通突发事件识别方法_刘昭



- [ ] 流程梳理

1. 随机选取500条，用来做数据预处理的数量即可`df.smaple()`
2. 文本预处理
   1. 文本去噪
   2. 中文分词
   3. 过滤停用词
   4. HanLP 词性标注 【没有特殊需求，咱们就不用做了
   5. 文本去重 【考虑发表的用户，**官方号**
   6. 文本标记【手动整理模型输入数据
   7. 特征词选择

3. 数据向量化，分类器模型可以直接使用



- [ ]  训练样本处理

- 对于训练集，自然语言处理过程包括：
  - 中文分词、过滤停用词、特征权重计算、特征词选取
- 而对于测试集：
  - 特征权重计算、特征词



## 具体代码实现

### 1.文本去噪

> 微博中的特殊符号，对于算法的训练来说不太有利
>
> [Python正则表达式清洗微博文本特殊符号(网址, @, 表情符等)](https://blog.csdn.net/blmoistawinde/article/details/103648044)
>
> [附：表达式全集（正则表达式手册）](https://blog.csdn.net/qq_33472765/article/details/80785441)

- 微博里主要有几种特殊格式：
  1. 网页
  2. @用户名（包括转发路径上的其他用户名）
  3. 表情符号(用[]包围)
  4. 话题 “#topic#”(用#包围)
  4. 分类标签“【content】”
  4. 删除字符数小于10 的过短微博
- 分析：
  - 1：没有用
  - 2：有时候会@官方号啥的，部分会有用，再考虑，删
  - 3：表情包里面也会有一些表情信息，[哈哈] ，太麻烦了，直接删
  - 4：这个就很重要的了，考虑保留【自己看了数据之后，发现里面很多废话
  - 5：标签里面，也有一些重要信息的，再考虑【自己看了数据之后，发现里面很多废话
- 疑问：
  - 话题两边的“#”要不要删掉，影响分词
  - 现在的代码“@”删的不彻底，还是会留下一些【后面有一个 \xa0 不间断空白符，已处理
  - 限制会把“@”之后的所有问题都删掉了？？？
  - 应用到DataFrame的每一个数据上map()函数
  - '⚠'没有被去掉？？？



### 2. 中文分词

> [结巴中文分词](https://github.com/fxsjy/jieba)

- 示例分析
  - 现在示例代码/文字只是依据单独的话，没有标点符号
  - 中英文的标点符号，也会被分词 - 后面过滤停用词的时候再删掉就好
  - 应用到DataFrame的每一个数据上map()函数
  - 分完词之后，把一次字的去掉？？？？

### 3.过滤停用词

> [Python-中文分词并去除停用词仅保留汉字](https://blog.csdn.net/lztttao/article/details/104723228)
>
> [python去除文本停用词（jieba分词+哈工大停用词表）](https://blog.csdn.net/weixin_39068956/article/details/116449126)
>
> [中文常用停用词表](https://github.com/goto456/stopwords)

- 获取停用词表 【stopwords/***.txt

  | 词表名 | 词表文件 |
  | - | - |
  | 中文停用词表          | cn\_stopwords.txt   |
  | 哈工大停用词表         | hit\_stopwords.txt  |
  | 百度停用词表          | baidu\_stopwords.txt |
  | 四川大学机器智能实验室停用词库 | scu\_stopwords.txt  |
  | 自定义词库 |  |



- 导入停用词库 *stopword_list*

- 直接 判断分好的词，是否在 *stopword_list*中

- 去除数字  [python 判断字符串的内容是不是数字](https://blog.csdn.net/m0_37622530/article/details/81289520)

  ```python
  def is_number(s):
      try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
          float(s)
          return True
      except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
          pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
      try:
          import unicodedata  # 处理ASCii码的包
          unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
          return True
      except (TypeError, ValueError):
          pass
      return False
  ```

- 一行for函数  `res23 =[w for w in res23 if not is_number(w)]`

### 4.词性标注 

- 没有这个需求

### 5.文本去重 

> 文本相似性分析：
>
> [python比较字符串相似度](https://blog.csdn.net/qq_41020281/article/details/82194992)



- df.duplicated('Class',*keep*='last') 直接判断重复
- df.drop_duplicates('Class')
- df.drop_duplicates(['School','Class'])
- df.sample(*n*=3,*weights*=df['Math'])

- 两条文本统一判断

  ```python
  import difflib
   
  def string_similar(s1, s2):
      return difflib.SequenceMatcher(None, s1, s2).quick_ratio()
   
  print string_similar('爱尔眼科沪滨医院', '沪滨爱尔眼科医院')
  print string_similar('安定区妇幼保健站', '定西市安定区妇幼保健站')
  print string_similar('广州市医院', '广东省中医院')
  print(string_similar("广州市医院 - 我和你","你和我 - 广州市医院"))
  
  out:
  1.0
  0.8421052631578947
  0.5454545454545454
  1.0
  ```

  - 需要将分好的词，再组合一下 `"".join(["地方","按时","放到"])`
  - 只要不是重复的，其实也不用去，就当作学习样本了，也可以
  - 

### 6.文本标记

- 手动处理

### 7.特征词选择

> 特征向量化
>
> - 基于微博数据的高速公路交通事件研究_包丹 【第三章 交通文本分类方法研究
> - 基于社交网络数据的交通突发事件识别方法_刘昭 【3.2 基于特征权重的特征词选取方法
> - [基于 TF-IDF 算法的关键词抽取](https://github.com/fxsjy/jieba)
> - [sklearn中的文本特征提取-英文](https://welts.xyz/2022/03/26/sklearn_text/)
> - [python文本特征提取词频矩阵](https://blog.csdn.net/weixin_39633781/article/details/112394825)





- 英文

```python
import pandas as pd
import numpy as np
from  sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

corpus = [
    "I have a dog.",
    "You have a dog and a cat.",
    "He books a book.",
    "No cost too great.",
]

# 使用 CountVectorizer
counter = CountVectorizer() # 先使用默认参数
counter.fit(corpus)
X = counter.transform(corpus)
print(counter.vocabulary_)
print(X.todense()) # X是一个稀疏矩阵，输出稠密化表示
df=pd.DataFrame(X.toarray(),columns=counter.get_feature_names_out())
print(df)

# 使用TfidfVectorizer
tfidf = TfidfVectorizer() # 先使用默认参数
X = tfidf.fit_transform(corpus)

df=pd.DataFrame(X.toarray(),columns=tfidf.get_feature_names_out())
print(df)
```

- 中文

```python
import pandas as pd
import numpy as np
from  sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

data=["移动共享，共享汽车，共享经济，共享单车",
     "财经栏目，财经政策，经济政策，共享经济"] 

# 需要提前分词
# 分词
cut_data=[]
for s in data:
    cut_s = jieba.cut(s)
    l_cut_s=' '.join(list(cut_s))    
    cut_data.append(l_cut_s)
    print(l_cut_s)
    
# 使用 CountVectorizer
transfer = CountVectorizer(stop_words=["打算","就是"]) 
#实例化一个转换器类,
# # stop_words=["打算","就是"],去除不想要的词
data_new = transfer.fit_transform(cut_data)  #调用fit_transform()
print(data_new)
print(transfer.get_feature_names_out())
print(data_new.toarray()) 
#构建成一个二维表：
df=pd.DataFrame(data_new.toarray(),columns=transfer.get_feature_names_out())
print(df)   

# TfidfVectorizer
transfer = TfidfVectorizer() #实例化一个转换器类
data_new = transfer.fit_transform(cut_data) #调用fit_transform()
print(data_new)
print(transfer.get_feature_names_out())
print(data_new.toarray()) 
#构建成一个二维表：
df=pd.DataFrame(data_new.toarray(),columns=transfer.get_feature_names_out())
print(df)
```

## 整合

- pandas 某列都应用吗，某一函数 函数有多参数

  > [给DataFrame的apply调用的函数添加多个参数](https://blog.csdn.net/jewely/article/details/107888098)

  ```python	
  # 前提：被调用的函数第一个参数必须是DataFrame的行或列
  
  df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]],
                    index=list('AB'),
                    columns=list('abcd'))
  print(df)
  '''
  	a	b	c	d
  A	1	2	3	4
  B	5	6	7	8
  '''
  
  def test(x, y, z):
      return x + y + z
  print(df.apply(test, args=(10, 100)))
  '''
       a    b    c    d
  A  111  112  113  114
  B  115  116  117  118
  '''
  
  print(df.apply(test, y=100, z=100))
  
  ```

  


## 存在问题分析

- 现在出行的词汇，排序的最高的，地名偏多，还需要结果具体问题考虑去停用词

  ```text
  ['鼓楼区', '鼓楼', '默默', '默兹河', '黔西南州', '黑车', '黄色', '黄河路', '鹤壁', '鱼塘', '魏城', '高铁', '高速公路', '高速', '高空槽', '高空作业', '高空', '高热量', '高温', '高栏', '高架', '高川', '高峰期', '高峰', '高处', '高位', '骤至', '骑着', '驾驶员', '驾驶', '驾车', '驻点', '驶离', '驱赶', '马路', '首报', '首义', '饱和', '食物', '飞仙', '风雨', '风险', '风景线', '风力', '频频', '频繁', '领导', '预防', '预计', '预警']
  ```

  

- 打标签，明天一起安排

  

- 说明

   1. 随机选了500条【每人可以打100条的标签
   1. df.sample(*n*=3,*weights*=df['Math'])
   2. 最后一步向量化，用了两种编码格式：TDIDF和CountVector，都可以直接用，看看哪个效果好吧（分别以TDIDF和CountVec结尾）【给DY看的
   3. 打标签只需要处理一个文件即可，“完整数据_暴雨_交通_pre_tweets_TFIDF_Label.xlsx”。两个编码格式处理的都是同样的500条数据。标签一次就好，打标签安排：
   
       - TH：0-199
   
       - ZRF：200-299
   
       - CYX：300-399
       - LJK：400-499
   4. 数据说明：相当于属性放在“完整数据_暴雨_交通_pre_tweets_TFIDF.cvx”里面了，数据我们放到excel里面【给DY看的



- 改进提升工作：

  - 很多不相关的词语，没有能够去掉
  - 自定义一些分词的**词库**
  - 自定义停用词的**词库**
  - 自定义一个关键词库，如果有这个特征，存在就给加权【相关的理论



- 会议讨论-2022/05/25

  - 句间关系 考虑	

  - 关键词选取的时候，构建特征向量的时候，还需要考虑词间的顺序

  - 
