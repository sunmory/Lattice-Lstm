# encoding: utf-8

import pickle as pkl
import tensorflow as tf
from LatticeLstm import LatticeLstm
from Crf import Crf

tf.enable_eager_execution()


class BiLstmCrf(tf.keras.Model):
    def __init__(self, args, char_num, kb_words_num, label_num):
        super(BiLstmCrf, self).__init__()
        self.args = args
        self.embed_size = self.args.embed_size
        self.kb_embedding_size = self.args.kb_embedding_size
        self.hidden_size = self.args.hidden_size
        self.learning_rate = self.args.learning_rate
        self.drop_rate = self.args.drop_rate
        self.label_num = label_num
        if self.args.char_embedding_path is None:
            self.char_embedding = tf.Variable(tf.random_uniform(shape=[char_num, self.embed_size]), trainable=True, dtype=tf.float32, name='char_embedding')
        else:
            with open(self.args.char_embedding_path, 'rb') as f:
                pre_char_embedding = pkl.load(f)
            self.char_embedding = tf.Variable(pre_char_embedding, trainable=False, dtype=tf.float32, name='char_embedding')
        if self.args.kb_embedding_path is None:
            self.kb_embedding = tf.Variable(tf.random_uniform(shape=[kb_words_num, self.kb_embedding_size]), trainable=True, dtype=tf.float32, name='kb_embedding')
        else:
            with open(self.args.kb_embedding_path, 'rb') as f:
                pre_kb_embedding = pkl.load(f)
            self.kb_embedding = tf.Variable(pre_kb_embedding, trainable=False, dtype=tf.float32, name='kb_embedding')

        self.forward_lattice_lstm = LatticeLstm(self.embed_size, int(self.hidden_size / 2), self.kb_embedding_size, self.drop_rate, left2right=True, use_biase=True)
        self.bakward_lattice_lstm = LatticeLstm(self.embed_size, int(self.hidden_size / 2), self.kb_embedding_size, self.drop_rate, left2right=False, use_biase=True)
        self.dense = tf.keras.layers.Dense(self.label_num, name='label_dense', bias_initializer=tf.zeros_initializer())
        self.crf = Crf(self.label_num)

    def call(self, char_ids, kb_word_ids, label, sequence_length, kb_words_length, decode=False):
        batch_size, batch_sentece_length = char_ids.shape[0], char_ids.shape[1]
        batch_lattice_output = []
        for sentence_index in range(char_ids.shape[0]):
            length = sequence_length[sentence_index]
            sentence_chr_embedding = tf.nn.embedding_lookup(self.char_embedding, char_ids[sentence_index, :])
            sentence_words_list = kb_word_ids[sentence_index]
            sentence_words_length = kb_words_length[sentence_index]
            assert len(sentence_words_list) == length
            sentence_words_embedding = []
            for word_list in sentence_words_list:
                if len(word_list) > 0:
                    sentence_words_embedding.append(tf.nn.embedding_lookup(self.kb_embedding, word_list))
                else:
                    sentence_words_embedding.append(None)
            forward_lattice_h, forward_lattice_c = self.forward_lattice_lstm(sentence_chr_embedding, sentence_words_embedding, sentence_words_length, length)
            bachword_lattice_h, bachword_lattice_c = self.bakward_lattice_lstm(sentence_chr_embedding, sentence_words_embedding, sentence_words_length, length)

            batch_lattice_output.append(tf.expand_dims(tf.concat([forward_lattice_h, bachword_lattice_h], axis=-1), axis=0))
        batch_lattice_output = tf.concat(batch_lattice_output, axis=0)
        batch_lattice_output = tf.reshape(batch_lattice_output, [-1, self.hidden_size])
        batch_lattice_output = self.dense(batch_lattice_output)
        batch_lattice_output = tf.reshape(batch_lattice_output, shape=[batch_size, batch_sentece_length, self.label_num])

        if not decode:
            loss = self.crf(batch_lattice_output, label, sequence_length)
            return loss
        else:
            predict_label = self.crf.decode(batch_lattice_output, sequence_length)
            return predict_label

