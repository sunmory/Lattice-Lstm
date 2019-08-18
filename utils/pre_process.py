# encoding: utf-8

import random
import json
import pickle as pkl
from tqdm import tqdm
from Trie import Trie
from model.args import parse_args


args = parse_args()


def strf2h(ostring):
    """
    全角转半角
    :param ostring:
    :return:
    """
    tstring = ''
    for ch in ostring:
        ch_code = ord(ch)
        if ch_code == 12288:
            ch_code = 32
        elif (ch_code >= 65281 and ch_code <= 65374):
            ch_code -= 65248
        tstring += chr(ch_code)
    return tstring


def str_lower(ostring):
    """
    英文字符小写
    :param ostring:
    :return:
    """
    tstring = ""
    for ch in ostring:
        tstring = tstring + ch.lower()
    return tstring


def generate_kb2id():
    id_dict = dict()
    id_dict['PAD'] = 0
    kb_word_id = 1
    trie = Trie()
    with open(args.kb_path, 'r', encoding='utf-8') as f:
        for word in f:
            word = word.strip()
            if word not in id_dict.keys():
                trie.insert(word)
                id_dict[word] = kb_word_id
                kb_word_id = kb_word_id + 1

    with open(args.trie_path, 'wb') as f:
        pkl.dump(trie, f)

    with open(args.kb2id_path, 'wb') as f:
        pkl.dump(id_dict, f)


def generate_ch2id(min_num=3):
    id_dict = dict()
    num_dict = dict()
    id_dict['PAD'] = 0
    id_dict['UNK'] = 1
    char_id_index = 2

    with open(args.train_path, 'r', encoding='utf-8') as f:
        for l in f:
            word_label = l.strip().split('\t')
            if len(word_label) == 1: continue
            assert len(word_label) == 2
            word = word_label[0]
            if word not in num_dict.keys():
                num_dict[word] = 1
            else:
                num_dict[word] = num_dict[word] + 1

    for word, num in num_dict.items():
        if num < min_num:
            continue
        id_dict[word] = char_id_index
        char_id_index = char_id_index + 1

    with open(args.cha2id_path, 'wb') as f:
        pkl.dump(id_dict, f)


def load_datatset(dataset_path):
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for l in f:
            dataset.append(json.loads(l))
    return dataset


def generate_train_dataset():
    train_path = '../Dataset/train.json'
    total_dataset = load_datatset(train_path)

    def write_dataset(dataset, mode='train'):
        if mode == 'train':
            dataset_path = '../Dataset/train.txt'
        if mode == 'dev':
            dataset_path = '../Dataset/dev.txt'
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for sentence_dict in tqdm(dataset):
                sentence = sentence_dict['text']
                sentence = strf2h(sentence)
                sentence = str_lower(sentence)
                labels = ['O'] * len(sentence)
                for mention in sentence_dict['mention_data']:
                    mention_name = mention['mention']
                    mention_index = int(mention['offset'])
                    mention_length = len(mention_name)
                    if mention_length == 1:
                        labels[mention_index] = 'S'
                    if mention_length == 2:
                        labels[mention_index] = 'B'
                        labels[mention_index + 1] = 'E'
                    if mention_length >= 3:
                        labels[mention_index] = 'B'
                        for label_index in range(mention_index + 1, mention_index + mention_length - 1):
                            labels[label_index] = 'I'
                        labels[mention_index + mention_length - 1] = 'E'
                for ch, label in zip(sentence, labels):
                    f.write(ch + '\t' + label + '\n')
                f.write('\n')

    dataset_index = list(range(len(total_dataset)))
    random.shuffle(dataset_index)

    train_dataset = [total_dataset[train_index] for train_index in dataset_index[:int(len(dataset_index) * 0.9)]]
    dev_dataset = [total_dataset[dev_index] for dev_index in dataset_index[int(len(dataset_index) * 0.9):]]

    write_dataset(train_dataset, mode='train')
    write_dataset(dev_dataset, mode='dev')


def generate_kb_dataset():
    kb_dataset_path = '../Dataset/kb_data'
    kb_dataset = load_datatset(kb_dataset_path)
    kb_words_set = set()
    for entity_dict in tqdm(kb_dataset):
        for alias_name in entity_dict['alias']:
            alias_name = strf2h(alias_name)
            alias_name = str_lower(alias_name)
            kb_words_set.add(alias_name)
        entity_name = entity_dict['subject']
        entity_name = strf2h(entity_name)
        entity_name = str_lower(entity_name)
        kb_words_set.add(entity_name)
    with open('../Dataset/kb.txt', 'w', encoding='utf-8') as f:
        for kb_word in kb_words_set:
            f.write(kb_word + '\r')


if __name__ == '__main__':
    # generate_train_dataset()
    # generate_kb_dataset()
    generate_ch2id()
    generate_kb2id()