import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords


class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility 是一个工具类，用于将原始HTML文本处理成片段"""

    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False):
        # 将文档转换为单词序列的函数，
        # 可选择是否移除停用词。返回一个单词列表。
        #
        # 1. 移除HTML标签
        review_text = BeautifulSoup(review, features="lxml").get_text()
        #
        # 2. 移除非字母字符
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
        #
        # 3. 将单词转换为小写并分割
        words = review_text.lower().split()
        #
        # 4. 可选择是否移除停用词（默认为false）
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. 返回单词列表
        return words

    # 定义一个函数将评论分割为解析后的句子
    @staticmethod
    def review_to_sentences(review, tokenizer, remove_stopwords=False):
        # 将评论分割为解析后的句子的函数。返回一个句子列表，
        # 每个句子是一个单词列表
        #
        # 1. 使用NLTK分词器将段落分割成句子
        raw_sentences = tokenizer.tokenize(review.strip())
        #
        # 2. 遍历每个句子
        sentences = []
        for raw_sentence in raw_sentences:
            # 如果句子为空，跳过
            if len(raw_sentence) > 0:
                # 否则，调用review_to_wordlist获取单词列表
                sentences.append(KaggleWord2VecUtility.review_to_wordlist(raw_sentence, \
                                                                          remove_stopwords))
        #
        # 返回句子列表（每个句子是一个单词列表，
        # 因此返回一个列表的列表）
        return sentences
