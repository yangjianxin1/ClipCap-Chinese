import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def statistics_flickr(file):
    """
    统计数据集的分布
    :param file:
    :return:
    """
    print('process file : {}'.format(file))
    with open(file, 'r') as f:
        lines = f.readlines()

    text_lens = []  # 统计所有句子的长度分布
    for index, line in tqdm(enumerate(lines)):
        line = line.split()
        caption = ''.join(line[1:])
        text_lens.append(len(caption))

    # 全部文本的长度分布
    plt.hist(x=text_lens,  # 指定绘图数据
             bins=20,  # 指定直方图中条块的个数
             color='steelblue',  # 指定直方图的填充色
             edgecolor='black',  # 指定直方图的边框色
             # weights=[1. / len(text_lens)] * len(text_lens)
             )
    plt.xlabel('sentence length')
    plt.ylabel('sentence count')
    plt.title('length distribution of data')
    plt.show()


if __name__ == '__main__':
    # file = 'datasets/train_caption.json'
    # statistics_coco(file)

    file = 'datasets/flickr_caption.txt'
    statistics_flickr(file)