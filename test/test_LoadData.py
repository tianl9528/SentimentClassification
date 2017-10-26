from collections import Counter
from src.SentimentNetwork import SentimentNetwork

from src.LoadData import LoadData

lable_file = '../datasets/labels.txt'
reviews_file = '../datasets/reviews.txt'

data = LoadData(reviews_file,lable_file)
reviews = data.reviews
labels = data.labels

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.05,learning_rate=0.01)
mlp.train(reviews[:-1500],labels[:-1500])

