from src.LoadData import LoadData
from src.SentimentNetwork import SentimentNetwork


labels_file = '../datasets/labels.txt'
reviews_file = '../datasets/reviews.txt'
data = LoadData(reviews_file,labels_file)


reviews = data.reviews
labels = data.labels

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.05,learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])
print('\n----------------------------------------------------')
mlp.test(reviews[-1000:],labels[-1000:])

