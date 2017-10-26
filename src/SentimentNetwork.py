from collections import Counter

import numpy as np
import time
import sys


class SentimentNetwork:
    def __init__(self, reviews, lablas, min_count=10, polarity_cutoff=0.1, hidden_nodes=10, learning_rate=0.1):
        # 设置随机因子
        np.random.seed(1)

        # 数据预处理
        self.pre_process_data(reviews, lablas, polarity_cutoff, min_count)

        # 初始化网络
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels, polarity_cutoff, min_count):
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
            if labels[i] == "positive":
                for word in reviews[i].split(' '):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(' '):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        pos_neg_ratios = Counter()

        for word, cnt in list(total_counts.most_common()):
            if cnt > 50:
                pos_neg_ratio = positive_counts[word] / float(negative_counts[word] + 1)
                pos_neg_ratios[word] = pos_neg_ratio

        for word, ratio in pos_neg_ratios.most_common():
            if ratio > 1:
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))

        review_vocab = set()
        for review in reviews:
            for word in review.split(' '):
                if total_counts[word] > min_count:
                    if word in pos_neg_ratios.keys():
                        if ((pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff)):
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)

        self.review_vocab = list(review_vocab)

        label_vocab = set()
        for label in labels:
            label_vocab.add(label)

        self.label_vocab = list(label_vocab)

        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        self.weights_1_2 = np.random.normal(0.0, self.output_nodes ** -0.5, (self.hidden_nodes, self.output_nodes))

        self.layer_1 = np.zeros((1, hidden_nodes))

    def get_target_for_label(self, label):
        if label == "positive":
            return 1
        else:
            return 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        return output * (1 - output)

    def train(self, train_reviews_raw, train_labels):
        train_reviews = list()
        for review in train_reviews_raw:
            indices = set()
            for word in review.split(' '):
                if word in self.word2index.keys():
                    indices.add(self.word2index[word])
            train_reviews.append(list(indices))

        assert len(train_reviews) == len(train_labels)

        corrent_so_far = 0

        start = time.time()

        for i in range(len(train_reviews)):

            review = train_reviews[i]
            label = train_labels[i]

            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]

            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            layer_2_error = layer_2 - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            layer_1_delta = layer_1_error

            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate

            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate

            if (layer_2 > 0.5 and label == 'positive'):
                corrent_so_far += 1
            elif (layer_2 < 0.5 and label == 'negative'):
                corrent_so_far += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write('\r进度:' + str(100 * i / float(len(train_reviews)))[:4] \
                             + '% \t速度(条/秒):' + str(reviews_per_second)[0:5] \
                             + '\t#正确预测数:' + str(corrent_so_far) \
                             + '\t#已训练:' + str(i + 1) \
                             + '\t训练准确率:' + str(corrent_so_far * 100 / float(i + 1))[0:4] + '%')
            if i % 5000 == 0:
                print('')

    def run(self, review):
        self.layer_1 *= 0

        unique_indices = set()
        for word in review.lower().split(' '):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]

        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

        if layer_2 > 0.5:
            return 'positive'
        elif layer_2 < 0.5:
            return 'negative'

    def test(self, testing_reviews, testing_labels):

        correct = 0

        start = time.time()

        for i in range(len(testing_reviews)):
            if self.run(testing_reviews[i]) == testing_labels[i]:
                correct += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write('\r进度:' + str(100 * i / float(len(testing_reviews)))[:4] \
                             + '% \t速度(条/秒):' + str(reviews_per_second)[0:5] \
                             + '\t#正确预测数:' + str(correct) \
                             + '\t#已测试:' + str(i + 1) \
                             + '\t测试准确率:' + str(correct * 100 / float(i + 1))[:4] + '%')
