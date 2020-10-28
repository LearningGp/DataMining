import gensim.downloader as api

from gensim.corpora import Dictionary

#加载数据 注意第一次加载需要下载 时间有点长

dataset = api.load("text8")

dct = Dictionary(dataset)

new_corpus = [dct.doc2bow(line) for line in dataset]

#加载模型库

from gensim import models

#训练模型

tfidf = models.TfidfModel(new_corpus)

#保存模型

tfidf.save("tfidf.model")

# 载入模型

tfidf = models.TfidfModel.load("tfidf.model")

# 使用这个训练好的模型得到单词的tfidf值

tfidf_vec = []

for i in range(len(new_corpus)):

    string_tfidf = tfidf[new_corpus[i]]

    tfidf_vec.append(string_tfidf)

# 输出 词语id与词语tfidf值

print(tfidf_vec)
