# encoding: utf-8

import tensorflow as tf

tf.enable_eager_execution()


class WordLstmCell(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, use_bias=True):
        super(WordLstmCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_i = tf.keras.layers.Dense(units=3 * self.hidden_size)
        self.weight_h = tf.keras.layers.Dense(units=3 * self.hidden_size)

        if self.use_bias:
            self.biase = self.add_variable(name='b',  shape=[3 * self.hidden_size], initializer=tf.random_uniform_initializer())

    def call(self, input, hx):
        """
        计算以句子当前字符起始的多个词汇的 word cell
        :param input: tensor
                word_num * word_embedding_size
        :param hx: list
                (h_o, c_o) ,((1 * hidden_size), (1 * hidden_size)), 当前字符的 h 与 c
        :return: tensor
                word_num * hidden_size , words cell
        """
        h_0, c_0 = hx
        wi = self.weight_i(input)
        wh = self.weight_h(h_0)

        wih = tf.add(wi, wh)

        if self.use_bias:
            wih = tf.add(wih, self.biase)
        f, i, g = tf.split(wih, 3, axis=1)
        c_1 = tf.sigmoid(f) * c_0 + tf.sigmoid(i) * tf.tanh(g)
        return c_1


class MultiInputLSTMCell(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, use_bias=True):
        super(MultiInputLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_i = tf.keras.layers.Dense(3 * self.hidden_size)
        self.weight_h = tf.keras.layers.Dense(3 * self.hidden_size)
        self.alpha_weight_i = tf.keras.layers.Dense(self.hidden_size)
        self.alpha_weight_h = tf.keras.layers.Dense(self.hidden_size)

        if self.use_bias:
            self.biase = self.add_variable(name='b', shape=[3 * self.hidden_size], initializer=tf.random_uniform_initializer())
            self.alpha_biase = self.add_variable(name='b', shape=[self.hidden_size], initializer=tf.random_uniform_initializer())

    def call(self, input, c_input, hx):
        """
        使用以当前字符结尾的多个词汇的 word cell 计算当前字符的 h 与 c
        :param input: tensor
                (1 * char_embedding_size)
        :param c_input: tensor
                (words_num, hidden_size)
        :param hx: list
                (h, c) ((1 * hidden_size), (1 * hidden_size))
        :return:
        """
        h_0, c_0 = hx
        wi = self.weight_i(input)
        wh = self.weight_h(h_0)
        wih = tf.add(wi, wh)

        if self.use_bias:
            wih = tf.add(wih, self.biase)
        i, o, g = tf.split(wih, 3, axis=1)
        i = tf.sigmoid(i)
        o = tf.tanh(o)
        g = tf.sigmoid(g)
        c_num = len(c_input)

        if c_num == 0:
            f = 1 - i
            c1 = f * c_0 + i * g
            h1 = o * tf.tanh(c1)
        else:
            c_input_tesnor = tf.concat(c_input, axis=0)
            alpha_wi = self.alpha_weight_i(input)

            if self.use_bias:
                alpha_wi = tf.add(alpha_wi, self.alpha_biase)
            alpha_wi = tf.tile(alpha_wi, [c_num, 1])
            alpha_wh = self.alpha_weight_h(c_input_tesnor)

            alpha = tf.sigmoid(alpha_wi + alpha_wh)
            alpha = tf.exp(tf.concat([i, alpha], axis=0))
            alpha_sum = tf.reduce_sum(alpha, 0)
            alpha = alpha / alpha_sum
            merge_c = tf.concat([g, c_input_tesnor], axis=0)
            c1 = merge_c * alpha
            c1 = tf.reshape(tf.reduce_sum(c1, axis=0), [1, -1])
            h1 = o * tf.tanh(c1)
        return h1, c1


class LatticeLstm(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, kb_emnedding_size, drop_rate, left2right=True, use_biase=True):
        super(LatticeLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kb_embedding_size = kb_emnedding_size
        self.drop_rate = drop_rate
        self.left2right = left2right
        self.use_bias = use_biase
        self.mlp_input_lstm = MultiInputLSTMCell(self.input_size, self.hidden_size, self.use_bias)
        self.word_lstm = WordLstmCell(self.kb_embedding_size, self.hidden_size, self.use_bias)

    def call(self, input, kb_embedding_list, kb_lengh_list, sentence_length):
        """

        :param input:
        :param kb_embedding_list:
        :param kb_lengh_list:
        :param sentence_length:
        :return:
        """
        batch_sentence_length = input.shape[0]
        if not self.left2right:
            kb_embedding_list, kb_lengh_list = convert_kb_list(kb_embedding_list, kb_lengh_list)
        hidden_output = []
        cell_output = []
        h = tf.Variable(tf.zeros([1, self.hidden_size]))
        c = tf.Variable(tf.zeros([1, self.hidden_size]))
        chr_index_list = list(range(sentence_length))
        input_c_list = init_list_of_objects(sentence_length)
        if not self.left2right:
            chr_index_list = chr_index_list[::-1]
        for chr_index in chr_index_list:
            single_char_embedding = tf.reshape(input[chr_index, :], [1, -1])
            (h, c) = self.mlp_input_lstm(single_char_embedding, input_c_list[chr_index], (h, c))
            hidden_output.append(h)
            cell_output.append(c)
            # match_num = len(kb_list[chr_index])
            if kb_embedding_list[chr_index] is None:
                match_num = 0
            else:
                match_num = kb_embedding_list[chr_index].shape[0]
            if match_num > 0:
                word_embedding = kb_embedding_list[chr_index]
                word_l = kb_lengh_list[chr_index]
                word_embedding = tf.nn.dropout(word_embedding, self.drop_rate)
                kb_words_c = self.word_lstm(word_embedding, (h, c))
                assert int(word_embedding.shape[0]) == int(kb_words_c.shape[0])
                for kb_word_index in range(match_num):
                    length = word_l[kb_word_index]
                    if self.left2right:
                        input_c_list[chr_index + length - 1].append(tf.reshape(kb_words_c[kb_word_index, :], [1, -1]))
                    else:
                        input_c_list[chr_index - length + 1].append(tf.reshape(kb_words_c[kb_word_index, :], [1, -1]))
        if not self.left2right:
            hidden_output = hidden_output[::-1]
            cell_output = cell_output[::-1]
        if len(hidden_output) < batch_sentence_length:
            hidden_output.extend([tf.zeros([1, self.hidden_size])] * (batch_sentence_length - sentence_length))
            cell_output.extend([tf.zeros([1, self.hidden_size])] * (batch_sentence_length - sentence_length))
        return tf.concat(hidden_output, axis=0), tf.concat(cell_output, axis=0)


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())
    return list_of_objects


def convert_kb_list(kb_embedding_list, kb_length_list):
    sen_len = len(kb_embedding_list)
    cov_kb_embedding = init_list_of_objects(sen_len)
    cov_kb_length = init_list_of_objects(sen_len)
    for char_index in range(sen_len):
        if kb_embedding_list[char_index] is not None:
            ch_word_indexs = list(range(kb_embedding_list[char_index].shape[0]))
            for word_index, word_length in zip(ch_word_indexs, kb_length_list[char_index]):
                word_embedding = kb_embedding_list[char_index][word_index, :]
                new_char_index = char_index + word_length - 1
                cov_kb_embedding[new_char_index].append(tf.reshape(word_embedding, [1, -1]))
                cov_kb_length[new_char_index].append(word_length)

    new_cov_kb = []
    for embedding_list in cov_kb_embedding:
        if len(embedding_list) > 0:
            new_cov_kb.append(tf.concat(embedding_list, 0))
        else:
            new_cov_kb.append(None)
    cov_kb_embedding = new_cov_kb

    return cov_kb_embedding, cov_kb_length
