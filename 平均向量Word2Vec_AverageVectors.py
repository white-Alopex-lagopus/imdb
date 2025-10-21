import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility


# ****** 定义创建平均词向量的函数
#

def makeFeatureVec(words, model, num_features):
    # 用于对给定段落中的所有词向量求平均值的函数
    #
    # 预初始化一个空的numpy数组（为了速度）
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word是一个包含模型词汇表中单词名称的列表
    # 将其转换为集合，以提高速度
    index2word_set = set(model.wv.index_to_key)
    #
    # 遍历评论中的每个单词，如果它在模型的词汇表中，
    # 则将其特征向量加到总和中
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model.wv[word])
    #
    # 将结果除以单词数量得到平均值
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # 给定一组评论（每个评论是一个单词列表），计算
    # 每个评论的平均特征向量并返回一个2D numpy数组
    #
    # 初始化计数器
    counter = 0.
    #
    # 预分配一个2D numpy数组（为了速度）
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # 遍历评论
    for review in reviews:
       #
       # 每处理1000个评论打印一次状态信息
       if counter % 1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       #
       # 调用上面定义的创建平均特征向量的函数
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)
       #
       # 递增计数器
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews


if __name__ == '__main__':

    # 从文件读取数据
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0, delimiter="\t", quoting=3)

    # 验证读取的评论数量（总共100,000条）
    print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n" % (len(train["review"]), len(test["review"]), len(unlabeled_train["review"])))

    # 加载punkt分词器
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # ****** 将标记和未标记的训练集分割成干净的句子
    #
    sentences = []  # 初始化一个空的句子列表

    print("Parsing sentences from training set")
    for review in train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    print("Parsing sentences from unlabeled set")
    for review in unlabeled_train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    # ****** 设置参数并训练word2vec模型
    #
    # 导入内置的日志记录模块并配置它，以便Word2Vec创建良好的输出消息
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # 设置各种参数的值
    num_features = 300    # 词向量维度
    min_word_count = 40   # 最小词频
    num_workers = 4       # 并行运行的线程数
    context = 10          # 上下文窗口大小
    downsampling = 1e-3   # 高频词的下采样设置

    # 初始化并训练模型（这将需要一些时间）
    print("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=num_workers, vector_size=num_features, min_count=min_word_count, window=context, sample=downsampling, seed=1)

    # 如果不打算进一步训练模型，调用init_sims将使模型更加内存高效
    model.wv.init_sims(replace=True)

    # 创建一个有意义的模型名称并保存模型以供后续使用是很有帮助的
    # 之后可以使用Word2Vec.load()加载它
    model_name = "300features_40minwords_10context"
    model.save(model_name)

    # 测试模型的一些功能
    if 'man' in model.wv and 'woman' in model.wv and 'child' in model.wv and 'kitchen' in model.wv:
        model.wv.doesnt_match("man woman child kitchen".split())
    if 'france' in model.wv and 'england' in model.wv and 'germany' in model.wv and 'berlin' in model.wv:
        model.wv.doesnt_match("france england germany berlin".split())
    if 'paris' in model.wv and 'berlin' in model.wv and 'london' in model.wv and 'austria' in model.wv:
        model.wv.doesnt_match("paris berlin london austria".split())
    if 'man' in model.wv:
        model.wv.most_similar("man")
    if 'queen' in model.wv:
        model.wv.most_similar("queen")
    if 'awful' in model.wv:
        model.wv.most_similar("awful")

    # ****** 为训练集和测试集创建平均向量
    #
    print("Creating average feature vecs for training reviews")

    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)

    print("Creating average feature vecs for test reviews")

    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)

    # ****** 将随机森林拟合到训练集，然后进行预测
    #
    # 使用100棵树将随机森林拟合到训练数据
    forest = RandomForestClassifier(n_estimators=100)

    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(trainDataVecs, train["sentiment"])

    # 测试并提取结果
    result = forest.predict(testDataVecs)

    # 写入测试结果
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
    print("Wrote Word2Vec_AverageVectors.csv")