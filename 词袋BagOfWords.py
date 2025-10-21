import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)

    print("The first review is:")
    print(train["review"][0])

    # input("Press Enter to continue...")

    print("Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window...")
    # nltk.download("stopwords")  # 下载文本数据集，包括停用词

    # 初始化一个空列表来保存清洗后的评论
    clean_train_reviews = []

    # 遍历每个评论；创建一个索引i，从0到电影评论列表的长度

    print("Cleaning and parsing the training set movie reviews...\n")
    for i in range(0, len(train["review"])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

    # ****** 从训练集创建词袋模型
    #
    print("Creating the bag of words...\n")

    # 初始化"CountVectorizer"对象，这是scikit-learn的词袋工具
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    # fit_transform()执行两个功能：首先，它拟合模型并学习词汇表；其次，它将训练数据转换为特征向量
    # fit_transform的输入应该是一个字符串列表
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy数组易于处理，因此将结果转换为数组
    np.asarray(train_data_features)

    # ******* 使用词袋训练随机森林
    #
    print("Training the random forest (this may take a while)...")

    # 初始化一个包含100棵树的随机森林分类器
    forest = RandomForestClassifier(n_estimators=100)

    # 将随机森林拟合到训练集，使用词袋作为特征，情感标签作为响应变量
    #
    # 这可能需要几分钟运行
    forest = forest.fit(train_data_features, train["sentiment"])

    # 创建一个空列表并逐个添加清洗后的评论
    clean_test_reviews = []

    print("Cleaning and parsing the test set movie reviews...\n")
    for i in range(0, len(test["review"])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

    # 为测试集获取词袋，并转换为numpy数组
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

    # 使用随机森林进行情感标签预测
    print("Predicting test labels...\n")
    result = forest.predict(test_data_features)

    # 将结果复制到pandas DataFrame，包含"id"列和"sentiment"列
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

    # 使用pandas写入逗号分隔的输出文件
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)
    print("Wrote results to Bag_of_Words_model.csv")