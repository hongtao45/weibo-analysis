# weibo-tweets-analysis



## 文本处理流程

> 代码：
>
> [data_preprocess_test.ipynb](./data_preprocess_test.ipynb)  【 全流程的学习和测试代码
>
> 参考文献：
>
> - 基于微博数据的高速公路交通事件研究_包丹
> - 基于社交网络数据的交通突发事件识别方法_刘昭



- [x] 流程梳理

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
  - 应用到DataFrame的每一个数据上apply()函数
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

- 获取停用词表 【05stopwords/***.txt

  | 词表名 | 词表文件 |
  | - | - |
  | 中文停用词表          | cn\_stopwords.txt   |
  | 哈工大停用词表         | hit\_stopwords.txt  |
  | 百度停用词表          | baidu\_stopwords.txt |
  | 四川大学机器智能实验室停用词库 | scu\_stopwords.txt  |
  | 自定义词库 | user_stopwords.txt |



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

  - **句间关系** 考虑	

  - 关键词选取的时候，构建特征向量的时候，还需要**考虑词间的顺序**

  - 





# weibo-tweets-analysis - Pro



## 文本处理流程

> 代码：
> [data_preprocess_pro_test.ipynb](./data_preprocess_pro_test.ipynb)  【 全流程的学习和测试代码-改进
>
> 参考文献：
> - [zhang17173/Event-Extraction](https://github.com/zhang17173/Event-Extraction)
> - 



- 文本预处理 **流程**（原来的一样，不做变化）
  1. 文本去噪
  2. 中文分词
  3. 过滤停用词
  4. 词性标注 【没有特殊需求，咱们就不用做了
  5. 文本去重 【考虑发表的用户，**官方号**
  6. 文本标记【手动整理模型输入数据
  7. 特征词选择



### 文本去噪

> [附：表达式全集（正则表达式手册）](https://blog.csdn.net/qq_33472765/article/details/80785441)

- 现有的问题：

  1. 话题两边的“#”要不要删掉，影响分词、分类标签“【content】”也删掉
  2. 现在的代码“@”删的不彻底，还是会留下一些【后面有一个 \xa0 不间断空白符，已处理
  3. 限制会把“@”之后的所有问题都删掉了？？？
  5. '⚠'没有被去掉？？？

  

- 问题解决：

  1. **未**解决，观察之后，删除
   - 【#1#  2  #3#】 ：只需要删除1、3 但实际也会删除2，测试之后发现，直接匹配的最外层的两个#号，就匹配一次
   - 思路：不用正则表达式，手动实现
     - 分类标签也是一样的
  
  2. 解决，发现了，就添加到函数里面
    - 解决，增加停止符号，不要全都删了
  3. 解决，手动记录和添加这些符号



### 中文分词

> [pkuseg：一个多领域中文分词工具包](https://github.com/lancopku/pkuseg-python)
>
> [中文句法分析及LTP使用](https://blog.csdn.net/asialee_bird/article/details/102610588) 【基本停止维护了，安装失败，放弃

- 功能

  - 支持细分领域

  - 直接有基于weibo的数据训练的模型

    

- 细领域训练及测试结果,以下是在不同数据集上的对比结果：

  | WEIBO  | Precision | Recall | F-score   |
  | ------ | --------- | ------ | --------- |
  | jieba  | 87.79     | 87.54  | 87.66     |
  | THULAC | 93.40     | 92.40  | 92.87     |
  | pkuseg | 93.78     | 94.65  | **94.21** |



- 模型参数记录

  ```
  pkuseg.pkuseg(model_name = "default", user_dict = "default", postag = False)
  	
  	model_name		模型路径。
  			        "default"，默认参数，表示使用我们预训练好的混合领域模型(仅对pip下载的用户)。
  				"news", 使用新闻领域模型。
  				"web", 使用网络领域模型。
  				"medicine", 使用医药领域模型。
  				"tourism", 使用旅游领域模型。
  			        model_path, 从用户指定路径加载模型。
  	user_dict		设置用户词典。
  				"default", 默认参数，使用我们提供的词典。
  				None, 不使用词典。
  				dict_path, 在使用默认词典的同时会额外使用用户自定义词典，可以填自己的用户词典的路径，词典格式为一行一个词（如果选择进行词性标注并且已知该词的词性，则在该行写下词和词性，中间用tab字符隔开）。
  	postag		        是否进行词性分析。
  				False, 默认参数，只进行分词，不进行词性标注。
  				True, 会在分词的同时进行词性标注。
  ```

  

- 可以使用的模型：

  从pip安装的用户在使用细领域分词功能时，只需要设置model_name字段为对应的领域即可，会自动下载对应的细领域模型。

  从github下载的用户则需要自己下载对应的预训练模型，并设置model_name字段为预训练模型路径。预训练模型可以在[release](https://github.com/lancopku/pkuseg-python/releases)部分下载。以下是对预训练模型的说明：

  - **news**: 在MSRA（新闻语料）上训练的模型。【**测试**
  - **web**: 在微博（网络文本语料）上训练的模型。【**测试**
  - **medicine**: 在医药领域上训练的模型。
  - **tourism**: 在旅游领域上训练的模型。
  - **mixed**: 混合数据集训练的通用模型。随pip包附带的是此模型。【**测试**

  我们还通过领域自适应的方法，利用维基百科的未标注数据实现了几个细领域预训练模型的自动构建以及通用模型的优化，这些模型目前仅可以在release中下载：

  - **art**: 在艺术与文化领域上训练的模型。
  - **entertainment**: 在娱乐与体育领域上训练的模型。
  - **science**: 在科学领域上训练的模型。
  - **default_v2**: 使用领域自适应方法得到的优化后的通用模型，相较于默认模型规模更大，但泛化性能更好。



- **待做**：计算各个分词模型的 分词效果
  - 如何评定分词效果

- 

### 过滤停用词

- 代码支持自定义新的停用词库

  | 词表名                         | 词表文件             |
  | ------------------------------ | -------------------- |
  | 中文停用词表                   | cn\_stopwords.txt    |
  | 哈工大停用词表                 | hit\_stopwords.txt   |
  | 百度停用词表                   | baidu\_stopwords.txt |
  | 四川大学机器智能实验室停用词库 | scu\_stopwords.txt   |
  | **自定义词库**                 | user_stopwords.txt   |

  

- 具体的实现，直接分到分词里面

  - 还是会有很多奇怪的符号：'⚠️', '🌪️', '⛈', '🌬️'  len() 之后的少部分长度为2，不好判断并删除
  
  - 自己加到`user_stopwords.txt`里
  
    


### 词性标注 

> [中文句法分析及LTP使用](https://blog.csdn.net/asialee_bird/article/details/102610588) 【安装失败，放弃
>
> [pkuseg: 一个多领域中文分词工具包](https://github.com/lancopku/pkuseg-python)
>
> [jieba: 结巴中文分词](https://github.com/fxsjy/jieba)



- 问题：
  - **后续可以考虑用词性标注之后，删除一些与研究主题不相关的词语**



- jieba 分词及词性标注

paddle模式词性标注对应表如下：

paddle模式词性和专名类别标签集合如下表，其中词性标签24 个（小写字母），专名类别标签4 个（大写字母）。

| 标签 | 含义     | 标签 | 含义     | 标签 | 含义     | 标签 | 含义     |
| ---- | -------- | ---- | -------- | ---- | -------- | ---- | -------- |
| n    | 普通名词 | f    | 方位名词 | s    | 处所名词 | t    | 时间     |
| nr   | 人名     | ns   | 地名     | nt   | 机构名   | nw   | 作品名   |
| nz   | 其他专名 | v    | 普通动词 | vd   | 动副词   | vn   | 名动词   |
| a    | 形容词   | ad   | 副形词   | an   | 名形词   | d    | 副词     |
| m    | 数量词   | q    | 量词     | r    | 代词     | p    | 介词     |
| c    | 连词     | u    | 助词     | xc   | 其他虚词 | w    | 标点符号 |



- pkuseg 分词及词性标注

| 标签 | 含义     | 标签 | 含义     | 标签 | 含义     |
| ---- | -------- | ---- | -------- | ---- | -------- |
| n    | 名词     | t    | 时间词   | s    | 处所词   |
| f    | 方位词   | m    | 数词     | q    | 量词     |
| b    | 区别词   | r    | 代词     | v    | 动词     |
| a    | 形容词   | z    | 状态词   | d    | 副词     |
| p    | 介词     | c    | 连词     | u    | 助词     |
| y    | 语气词   | e    | 叹词     | o    | 拟声词   |
| i    | 成语     | l    | 习惯用语 | j    | 简称     |
| h    | 前接成分 | k    | 后接成分 | g    | 语素     |
| x    | 非语素字 | w    | 标点符号 | nr   | 人名     |
| ns   | 地名     | nt   | 机构名称 | nx   | 外文字符 |
| nz   | 其它专名 | vd   | 副动词   | vn   | 名动词   |
| vx   | 形式动词 | ad   | 副形词   | an   | 名形词   |





### 文本去重 

- 计算相似度

  ```python
  def vector_similarity(s1, s2):
      '''
      计算两个句子之间的相似度:将两个向量的夹角余弦值作为其相似度
      :param s1:
      :param s2:
      :return:
      '''
      v1, v2 = sentence_vector(s1), sentence_vector(s2)
      return np.dot(v1, v2) / (linalg.norm(v1) * linalg.norm(v2))
  ```

  
  
- 不同地方出现的微博在打标签时，不同人处理，可能会有不同的标签

- 在标记前之前，先标记重复文本，保证结果统一，同时减少工作量

  

### 文本标记

- 增加多个标签信息

  - 该条微博，是否包含了，与极端天气下的交通有关的信息（identify weather info
  - 该条微博，是否包含了，与道路损坏等有关的信息（identify road info

- 现有的五百条数据，再补充一个标签

  

### 特征词选择

- 向量化之**前**进行处理
  - 根据词性删除一些词语
  - 根据词性保留一些词语
- 向量化之**后**进行处理
  - 根据各词语在整体中出现的频率（权重）降维
  - 使用算法进行降维（不考虑数据的实际意义），例如主成分分析
  - 
  
- 向量化的方法
  - TFIDF
  - CounterVector
  - BERT【学习ing
  - word2vec【学习ing



# weibo-tweets-analysis - Pro Max



