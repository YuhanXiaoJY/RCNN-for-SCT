# -*- coding: utf-8 -*-
import skipthoughts
import pickle
import csv
from tqdm import tqdm
import numpy as np
train_raw = 'E:/Life_in_PKU/大三上/NLP导论/project/自然语言处理导论作业/故事结尾预测/data/rawdata/train.csv'.decode(encoding='utf-8')
train_preproc = '../processed_data/train_preproc.npy'
val_raw = 'E:/Life_in_PKU/大三上/NLP导论/project/自然语言处理导论作业/故事结尾预测/data/rawdata/val.csv'.decode(encoding='utf-8')
val_preproc = '../processed_data/val_preproc.npy'
test_raw = 'E:/Life_in_PKU/大三上/NLP导论/project/自然语言处理导论作业/故事结尾预测/data/rawdata/test.csv'.decode(encoding='utf-8')
test_preproc = '../processed_data/test_preproc.npy'


def get_text():
    def train_handler():
        file = open(train_raw, 'r')
        lines = csv.reader(file)

        index_rows = []
        for (num, oldline) in tqdm(enumerate(lines)):
            if num == 0:
                continue

            line = []
            for sentc in oldline:
                ch_list = [ch for ch in sentc if ch not in puncs]
                sentc = ''.join(ch_list)
                line.append(sentc)

            index_row = encoder.encode(line)
            index_rows.append(index_row)
        file.close()

        np.save(train_preproc, np.array(index_rows))

    def val_handler():
        print '[val_handler]: start to encode texts'
        file = open(val_raw, 'r')
        lines = csv.reader(file)

        index_rows = []
        for (num, oldline) in tqdm(enumerate(lines)):
            if num == 0:
                continue
            # remove the punctuations
            line = []
            for sentc in oldline:
                ch_list = [ch for ch in sentc if ch not in puncs]
                sentc = ''.join(ch_list)
                line.append(sentc)

            index_row = encoder.encode(line[:-1])
            index_rows.append(index_row)
        file.close()

        np.save(val_preproc, np.array(index_rows))
        print '[val_handler]: texts encoded'

    def test_handler():
        print '[test_handler]: start to encode texts'
        file = open(test_raw, 'r')
        lines = csv.reader(file)

        index_rows = []
        for (num, oldline) in tqdm(enumerate(lines)):
            if num == 0:
                continue

            line = []
            for sentc in oldline:
                ch_list = [ch for ch in sentc if ch not in puncs]
                sentc = ''.join(ch_list)
                line.append(sentc)

            index_row = encoder.encode(line)
            index_rows.append(index_row)
        file.close()

        np.save(test_preproc, np.array(index_rows))
        print '[test_handler]: texts encoded'

    puncs = '()*+,-./:;<=>?@[\]^_`{|}~!"#$%&'
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    #train_handler()
    val_handler()
    test_handler()



if __name__ == "__main__":
    get_text()
