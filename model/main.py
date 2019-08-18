# encoding: utf-8

import os
import copy
import datetime
import pickle as pkl
import numpy as np
import tensorflow as tf
from BilstmCrf import BiLstmCrf
from model.args import parse_args
from utils.Trie import Trie
from tqdm import tqdm

args = parse_args()

with open(args.cha2id_path, 'rb') as f:
    char2ids = pkl.load(f)
    
with open(args.kb2id_path, 'rb') as f:
    kb2ids = pkl.load(f)

label2ids = {'B': 0, 'I': 1, 'O': 2, 'E': 3, 'S': 5}


def generate_trie():
    trie = Trie()
    with open(args.kb_path, 'r', encoding='utf-8') as f:
        for word in f:
            word = word.strip()
            trie.insert(word)

    return trie


trie = generate_trie()


def read_dataset(mode='train'):
    if mode == 'train':
        data_path = args.train_path
    if mode == 'dev':
        data_path = args.dev_path
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset, label = [], []
        sentence, sentence_label = [], []
        for l in tqdm(f):
            w_l = l.split('\t')
            try:
                assert len(w_l) == 2 or len(l.strip()) == 0
            except AssertionError as e:
                print('skip line: {}'.format(l))
                continue
            if len(w_l) == 2:
                sentence.append(w_l[0])
                sentence_label.append(w_l[1].strip())
            else:
                dataset.append(sentence)
                label.append(sentence_label)
                sentence, sentence_label = [], []
    return dataset, label


def batch_genarator(mode='train'):
    dataset, label = read_dataset(mode)
    if mode == 'dev':
        dataset = dataset[:1000]
    batch_size = args.batch_size
    for index in tqdm(list(range(0, len(dataset), batch_size))):
        yield dataset[index: index + batch_size], label[index: index + batch_size]


def match_kb_words(sentences):
    kb_match_words = []
    kb_match_length = []
    for sentence in sentences:
        sentence_match_words = []
        sentence_match_length = []
        for char_index in range(len(sentence)):
            match_words = trie.enumerateMatch(sentence[char_index:])
            sentence_match_words.append(match_words)
            sentence_match_length.append([len(w) for w in match_words])
        kb_match_words.append(sentence_match_words)
        kb_match_length.append(sentence_match_length)
    return kb_match_words, kb_match_length


def padding(dataset, label):
    max_sentence_length = 0
    dataset_length, kb_matched_num = [], []
    for sentence in dataset:
        if len(sentence) > max_sentence_length:
            max_sentence_length = len(sentence)
    for sentence_index in range(len(dataset)):
        dataset_length.append(len(dataset[sentence_index]))
        dataset[sentence_index].extend(['PAD'] * (max_sentence_length - len(dataset[sentence_index])))
        label[sentence_index].extend(['O'] * (max_sentence_length - len(label[sentence_index])))
    return dataset, label, dataset_length


def change2id(dataset, kb_match_words, label):
    for sentence_index in range(len(dataset)):
        sentence_ch_ids = [char2ids.get(ch, 1) for ch in dataset[sentence_index]]
        sentence_kb_wds = [[kb2ids.get(w) for w in w_l] for w_l in kb_match_words[sentence_index]]
        sentence_label_ids = [label2ids[l] for l in label[sentence_index]]
        dataset[sentence_index] = sentence_ch_ids
        kb_match_words[sentence_index] = sentence_kb_wds
        label[sentence_index] = sentence_label_ids

    return np.array(dataset), kb_match_words, np.array(label)


def train(epoch):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    model_path = os.path.join(args.model_path, time_str + "/" + 'lattice_model')
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    for e in range(epoch):
        dataset_generator = batch_genarator(mode='train')
        batch_index, best_f1 = 0, 0
        try:
            while True:
                dataset, label = next(dataset_generator)
                kb_match_words, kb_match_length = match_kb_words(dataset)
                dataset, label, dataset_length = padding(dataset, label)
                dataset, kb_match_words, label = change2id(dataset, kb_match_words, label)
                with tf.GradientTape() as tape:
                    loss = lattice_bilstm(dataset, kb_match_words, label, dataset_length, kb_match_length)
                    loss = tf.reduce_mean(loss)
                grads = tape.gradient(loss, lattice_bilstm.trainable_variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, lattice_bilstm.trainable_variables))

                print('epoch: {}, batch_index: {}, loss: {}'.format(e, batch_index, loss))
                if (batch_index + 1) % 50 == 0:
                    p, r, f1 = evaluate()
                    print('epoch: {}, batch_index: {}, P: {}, R: {}, F1: {}, best_F1: {}'.format(e, batch_index, p, r, f1, best_f1))

                    if f1 > best_f1:
                        print('save model')
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        lattice_bilstm.save_weights(model_path)
                        best_f1 = f1
                batch_index = batch_index + 1
        except StopIteration as e:
            print('finish epoch {}'.format(e))


def compute_evaluate_score(predict_label, label):
    p_n, r_n, c_n = 0, 0, 0

    def get_entity_set(l):
        entity_set = set()
        BE_set = set()
        l_length = len(l)
        for index in range(l_length):
            if l[index] is 'B' and len(BE_set) is 0:
                BE_set.add(index)
            if l[index] is 'E' and len(BE_set) is 1:
                BE_set.add(index)
                entity_set.add(tuple(BE_set))
                BE_set = set()
            if l[index] is 'S':
                entity_set.add(index)
        return entity_set

    for pl, rl in zip(predict_label, label):
        pl_set = get_entity_set(pl)
        rl_set = get_entity_set(rl)
        p_n = p_n + len(pl_set)
        r_n = r_n + len(rl_set)
        c_n = c_n + len(pl_set.intersection(rl_set))

    return p_n, r_n, c_n


def evaluate():
    dataset_generator = batch_genarator(mode='dev')
    total_p_n, total_r_n, total_c_n = 0, 0, 0
    try:
        while True:
            o_dataset, o_label = next(dataset_generator)
            kb_match_words, kb_match_length = match_kb_words(o_dataset)
            dataset, label, dataset_length = padding(copy.deepcopy(o_dataset), copy.deepcopy(o_label[:]))
            dataset, kb_match_words, label = change2id(dataset, kb_match_words, label)
            predict_labels = lattice_bilstm(dataset, kb_match_words, label, dataset_length, kb_match_length, decode=True)
            p_n, r_n, c_n = compute_evaluate_score(predict_labels, o_label)
            total_p_n = total_p_n + p_n
            total_r_n = total_r_n + r_n
            total_c_n = total_c_n + c_n
    except StopIteration as e:
        print('finish evaluate')
    if total_p_n is 0 or total_r_n is 0 or total_c_n is 0:
        return 0
    p = total_c_n / total_p_n
    r = total_c_n / total_r_n
    f1 = 2 * p * r / (p + r)
    return p, r, f1


if __name__ == '__main__':
    lattice_bilstm = BiLstmCrf(args, len(char2ids.keys()), len(kb2ids.keys()), len(label2ids.keys()))
    train(3)