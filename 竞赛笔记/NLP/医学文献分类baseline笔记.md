## 任务目标

判断该文献是否属于医学文献,分类

### 训练集

- uuid:唯一标识符
- title:论文标题
- author:论文作者
- abstract:论文摘要
- Keywords:关键词
- label:标记是否为ai生成的

### 简要分析


#### Bow词袋模型

使用的是Bow词袋模型，是自然语言处理和文本挖掘中常用的一种基础特征表示方法。它是一种简单而有效的方法，用于将文本转换成数值特征向量，以便在机器学习模型中进行处理。

BOW模型的基本思想是将文本视为一组无序的单词（词汇表中的词），并将文本中每个单词的出现与否作为一个特征，构成一个向量，其中向量的每个维度表示一个词。在BOW模型中，不考虑单词在文本中的顺序和结构，只关注每个词在文本中的出现频率或者是否存在。

BOW模型的步骤：

- 构建词汇表（Vocabulary）：将所有文本数据中出现的不同单词收集起来，构建一个词汇表。每个不同的单词都作为词汇表中的一个词。

- 向量化：对于每个文本，创建一个与词汇表长度相同的向量，其中每个维度对应一个词汇表中的单词。向量的值可以表示该单词在文本中的出现次数、出现与否等。

BOW模型的优点是简单易用，能够捕捉到文本中不同词汇的存在与否，适用于一些简单的文本分类和情感分析等任务。然而，它忽略了词汇的顺序信息和语义关系，无法捕捉更复杂的语义含义。

而朵朵上一次NLP竞赛使用的是 TF-IDF模型，这两个模型是类似的，用统计学的思路去进行语句分析，但是问题就在于它**并不了解语义是什么**！

如果能采用深度学习模型，那么应该效果可以达到更好！

**单词需要转化为向量才能被机器学习算法采用！**

#### Baseline 0.67116
```py
# 导入pandas用于读取表格数据
import pandas as pd

# 导入BOW（词袋模型），可以选择将CountVectorizer替换为TfidfVectorizer（TF-IDF（词频-逆文档频率）），注意上下文要同时修改，亲测后者效果更佳
from sklearn.feature_extraction.text import CountVectorizer

# 导入LogisticRegression回归模型
from sklearn.linear_model import LogisticRegression

# 过滤警告消息
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


# 读取数据集
train = pd.read_csv('/home/aistudio/data/data231041/train.csv')
train['title'] = train['title'].fillna('')
train['abstract'] = train['abstract'].fillna('')

test = pd.read_csv('/home/aistudio/data/data231041/testB.csv')
test['title'] = test['title'].fillna('')
test['abstract'] = test['abstract'].fillna('')


# 提取文本特征，生成训练集与测试集
train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')
test['text'] = test['title'].fillna('') + ' ' +  test['author'].fillna('') + ' ' + test['abstract'].fillna('')

vector = CountVectorizer().fit(train['text'])
train_vector = vector.transform(train['text'])
test_vector = vector.transform(test['text'])


# 引入模型
model = LogisticRegression()

# 开始训练，这里可以考虑修改默认的batch_size与epoch来取得更好的效果
model.fit(train_vector, train['label'])

# 利用模型对测试集label标签进行预测
test['label'] = model.predict(test_vector)
test['Keywords'] = test['title'].fillna('')
test[['uuid','Keywords','label']].to_csv('submit_task1.csv', index=None)
```

#### 可能的优化空间

- 尝试TF-IDF或者选择不同的机器学习模型
- 调整词袋模型参数
- 重新分割训练集和测试集
- 再次尝试使用深度学习模型（好像不知道怎么写）