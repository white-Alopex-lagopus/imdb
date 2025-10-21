from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
import os
from KaggleWord2VecUtility import KaggleWord2VecUtility


# 定义创建质心词袋的函数
#
def create_bag_of_centroids(wordlist, word_centroid_map):
    #
    # 聚类数量等于词/质心映射中的最高聚类索引加1
    num_centroids = max(word_centroid_map.values()) + 1
    #
    # 预分配质心词袋向量（为了速度）
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    #
    # 遍历评论中的单词。如果单词在词汇表中，
    # 找到它属于哪个聚类，并将该聚类计数加一
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # 返回"质心词袋"
    return bag_of_centroids


if __name__ == '__main__':

    model = Word2Vec.load("300features_40minwords_10context")

    # ****** 在词向量上运行k-means并打印一些聚类
    #

    start = time.time()  # 开始时间

    # 将"k"（聚类数量）设置为词汇表大小的1/5，或者每个聚类平均5个单词
    word_vectors = model.wv.vectors
    num_clusters = word_vectors.shape[0] // 5

    # 初始化k-means对象并使用它提取质心
    print("Running K means")
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    # 获取结束时间并打印过程耗时
    end = time.time()
    elapsed = end - start
    print("Time taken for K Means clustering: ", elapsed, "seconds.")

    # 创建词/索引字典，将每个词汇词映射到聚类编号
    word_centroid_map = dict(zip(model.wv.index_to_key, idx))

    # 打印前十个聚类
    for cluster in range(0, 10):
        #
        # 打印聚类编号
        print("\nCluster %d" % cluster)
        #
        # 找到该聚类编号的所有单词，并打印出来
        words = []
        for word, centroid_id in word_centroid_map.items():
            if centroid_id == cluster:
                words.append(word)
        print(words)

    # 像之前一样创建clean_train_reviews和clean_test_reviews
    #

    # 从文件读取数据
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3)

    print("Cleaning training reviews")
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))

    print("Cleaning test reviews")
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))

    # ****** 创建质心词袋
    #
    # 为训练集质心词袋预分配数组（为了速度）
    train_centroids = np.zeros((len(train["review"]), num_clusters), dtype="float32")

    # 将训练集评论转换为质心词袋
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    # 对测试评论重复此过程
    test_centroids = np.zeros((len(test["review"]), num_clusters), dtype="float32")

    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    # ****** 拟合随机森林并提取预测
    #
    forest = RandomForestClassifier(n_estimators=100)

    # 拟合森林可能需要几分钟
    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(train_centroids, train["sentiment"])
    result = forest.predict(test_centroids)

    # 写入测试结果
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("BagOfCentroids.csv", index=False, quoting=3)
    print("Wrote BagOfCentroids.csv")