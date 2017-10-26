
class LoadData:
    def __init__(self, reviews_file, labels_file):
        g = open(reviews_file, 'r')
        self.reviews = list(map(lambda x:x[:-1], g.readlines()))
        g.close()
        g = open(labels_file, 'r')
        self.labels = list(map(lambda x: x[:-1], g.readlines()))
        g.close()

    # 输出标签和评论
    def show(self,i):
        print(self.labels[i] + '\t:\t' + self.reviews[i][:80] + '…')

    # 数据量
    def num(self):
        print(len(self.reviews))

    # 输出评论
    def show_review(self, i):
        print(self.reviews[i])

    # 输出标签
    def show_labal(self, i):
        print(self.labels[i])

