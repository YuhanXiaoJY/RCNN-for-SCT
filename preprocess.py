from data_util import *
import csv


def load_embedding_src():
    """
    no parameters
    :return: A dictionary. The key is a word and the value refers to its index to the embeddings
    """
    print('[preprocess]: building dictionary.')
    file = open(FLAGS.embedding_src1, 'r', encoding='utf-8')
    word_dict = {}
    embeddings = []
    index = 0
    line = file.readline()
    while line:
        line = line.strip().split(' ')
        word_dict[line[0]] = index
        embedding = [float(num) for num in line[1:]]
        embeddings.append(embedding)
        index += 1
        line = file.readline()

    file.close()
    # np.save('data/embedding/Glove/embeddings.npy', np.array(embeddings))
    print('[preprocess]: dictionary built.')
    return word_dict


def word2index():
    """
    build the index of raw data to the Glove embedding matrix
    :return: index lists
    """
    puncs = '()*+,-./:;<=>?@[\]^_`{|}~!"#$%&'
    def train_handler():
        for i in range(sentc_cnt):
            ch_list = [ch for ch in line[i] if ch not in puncs]
            line[i] = ''.join(ch_list)
            words = line[i].split(' ')
            words = [word.replace(' ', '') for word in words]
            words = [word.lower() for word in words]
            try:
                sentc = []
                for word in words:
                    if word in word_dict.keys():
                        sentc.append(word_dict[word])
                    else:
                        sentc.append(word_dict['<unk>'])
                sentcs.append(sentc)
            except:
                print(words)
    def val_handler():
        for i in range(sentc_cnt - 1):
            ch_list = [ch for ch in line[i] if ch not in puncs]
            line[i] = ''.join(ch_list)
            words = line[i].split(' ')
            words = [word.replace(' ', '') for word in words]
            words = [word.lower() for word in words]
            try:
                sentc = []
                for word in words:
                    if word in word_dict.keys():
                        sentc.append(word_dict[word])
                    else:
                        sentc.append(word_dict['<unk>'])
                sentcs.append(sentc)
            except:
                print(words)

    def test_handler():
        for i in range(sentc_cnt):
            ch_list = [ch for ch in line[i] if ch not in puncs]
            line[i] = ''.join(ch_list)
            words = line[i].split(' ')
            words = [word.replace(' ', '') for word in words]
            words = [word.lower() for word in words]
            try:
                sentc = []
                for word in words:
                    if word in word_dict.keys():
                        sentc.append(word_dict[word])
                    else:
                        sentc.append(word_dict['<unk>'])
                sentcs.append(sentc)
            except:
                print(words)

    word_dict = load_embedding_src()

    name_dict = {FLAGS.train_raw: FLAGS.train_embedding1, FLAGS.val_raw: FLAGS.val_embedding1, FLAGS.test_raw: FLAGS.test_embedding1}
    for filename in name_dict.keys():
        word_index = []
        file = open(filename, 'r', encoding='utf-8')
        lines = csv.reader(file)
        labels = []

        flag = False
        for (num, line) in enumerate(lines):
            # skip the first line
            if flag == False:
                flag = True
                continue

            sentcs = []
            sentc_cnt = len(line)
            if filename == FLAGS.train_raw:
                train_handler()
            elif filename == FLAGS.val_raw:
                val_handler()
                labels.append(line[6])
            elif filename == FLAGS.test_raw:
                test_handler()

            word_index.append(sentcs)
        file.close()

        # save the index list
        output = open(name_dict[filename], 'wb')
        pickle.dump(word_index, output)
        output.close()

        if filename == FLAGS.val_raw:
            label_file = open(FLAGS.val_labels, 'wb')
            pickle.dump(labels, label_file)
            label_file.close()


if __name__ == '__main__':
    pass
    #load_embedding_src()
    # get2016data()