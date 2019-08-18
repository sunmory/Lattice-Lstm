# encoding: utf-8

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prepare',
        action='store_true',
        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', default=False, help='train the model')
    parser.add_argument('--dev', default=True, help='evaluate the model on dev set')
    parser.add_argument('--predict', default=False, help='predict the answers for test set with trained model')
    parser.add_argument("--eopch", type=int, default=10, help="eopch")
    parser.add_argument("--embed_size", type=int, default=300, help="embed_size")
    parser.add_argument("--kb_embedding_size", type=int, default=300, help="kb_embedding_size")
    parser.add_argument("--hidden_size", type=int, default=300, help="hidden_size")
    parser.add_argument("--batch_size", type=int, default=100, help="batch_size")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop_rate")
    parser.add_argument("--use_biase", type=bool, default=True, help="use_biase")
    parser.add_argument('--train_path', type=str, default='../Dataset/train.txt', help='train_path')
    parser.add_argument('--dev_path', type=str, default='../Dataset/dev.txt', help='dev_path')
    parser.add_argument('--kb_path', type=str, default='../Dataset/kb.txt', help='kb_path')
    parser.add_argument('--char_embedding_path', type=str, default=None, help='embedding_path')
    parser.add_argument('--kb_embedding_path', type=str, default=None, help='embedding_path')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--trie_path', type=str, default='../Dataset/trie.pkl', help='trie_path')
    parser.add_argument('--cha2id_path', type=str, default='../Dataset/char2id.pkl', help='cha2id_path')
    parser.add_argument('--kb2id_path', type=str, default='../Dataset/kb_word2id.pkl', help='kb2id_path')
    parser.add_argument('--model_path', type=str, default='../output', help='model_path')
    args = parser.parse_args()

    return args