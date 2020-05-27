import numpy as np
import os
from collections import Counter
import tensorflow as tf
import sys

sys.path.append("../ELMo_self_train")

from bilm import (
    TokenBatcher,
    BidirectionalLanguageModel,
    weight_layers,
    dump_token_embeddings,
)


def read_data(filename):
    entity_list, token_list, tag_list = [], [], []
    entity, token_seq, tag_seq = [], [], []
    with open(filename, "r", encoding="utf8") as fr:
        for line in fr:
            ll = line.rstrip().split("\t")
            if len(ll) != 2:  # 每条数据中间用换行分隔
                assert len(token_seq) == len(tag_seq)
                if entity and token_seq and tag_seq:
                    entity_list.append(entity)
                    token_list.append(token_seq)
                    tag_list.append(tag_seq)
                entity, token_seq, tag_seq = [], [], []
            else:
                token, tag = ll
                if tag == "<head>":
                    if entity:
                        raise ValueError
                    else:
                        entity = list(token.split("#")[-1])
                else:
                    token_seq.append(token)
                    tag_seq.append(tag)
    return entity_list, token_list, tag_list


def get_vocab(train_file):
    entity_list, token_list, tag_list = read_data(train_file)
    token_set, tag_set = [], []

    for tmp in entity_list + token_list:
        token_set.extend(tmp)
    counter = Counter(token_set)  # 从大到小排序
    # count_pairs = counter.most_common(int(len(counter) * 0.8))
    count_pairs = [(k, v) for k, v in counter.items() if v >= 2]  # 去除频率低于3的词
    words, _ = list(zip(*count_pairs))
    words = ["<BLANK>", "<UNK>"] + list(words)
    word_dict = dict(zip(words, range(len(words))))

    for tmp in tag_list:
        tag_set.extend(tmp)
    tag_set = set(tag_set)

    ner_categories = list(sorted(set([x.split("_")[0] for x in tag_set])))
    classifier_categories = ["True", "False"]
    ner_cat2id_dict = dict(zip(ner_categories, range(len(ner_categories))))
    classifier_cat2id_dict = dict(
        zip(classifier_categories, range(len(classifier_categories)))
    )

    return (
        word_dict,
        [ner_categories, ner_cat2id_dict],
        [classifier_categories, classifier_cat2id_dict],
    )


def load_tag_vec(filename, ner_cat2id_dict, classifier_cat2id_dict, seq_len):
    _, _, tag_list = read_data(filename)
    ner_num_classes = len(ner_cat2id_dict)
    classifier_num_classes = len(classifier_cat2id_dict)

    ner_vec_list, classifier_vec_list = [], []
    for index in range(len(tag_list)):
        ner_tag_vec, classifier_tag_vec = [], []
        for tag in tag_list[index][:seq_len]:
            ner_tag = tag.split("_")[0]
            classifier_tag = "True" if "True" in tag else "False"
            if ner_tag in ner_cat2id_dict:
                vec = [0] * ner_num_classes
                vec[ner_cat2id_dict[ner_tag]] = 1
                ner_tag_vec.append(vec)
                vec = [0] * classifier_num_classes
                vec[classifier_cat2id_dict[classifier_tag]] = 1
                classifier_tag_vec.append(vec)
            else:
                print(tag)
                raise ValueError

        # 对标签序列进行padding
        for _ in range(len(ner_tag_vec), seq_len):
            vec = [0] * ner_num_classes
            vec[ner_cat2id_dict["O"]] = 1
            ner_tag_vec.append(vec)
            vec = [0] * classifier_num_classes
            vec[classifier_cat2id_dict["False"]] = 1
            classifier_tag_vec.append(vec)

        ner_vec_list.append(ner_tag_vec)
        classifier_vec_list.append(classifier_tag_vec)

    # 按每条数据打包
    labels = []
    for index in range(len(ner_vec_list)):
        labels.append([ner_vec_list[index], classifier_vec_list[index]])
    return np.array(labels)


def load_ELMo_data(filename, seq_len, entity_len):
    vocab_file = "./ELMo_file/vocab.txt"
    batcher = TokenBatcher(vocab_file)
    entity_list, token_list, _ = read_data(filename)

    entity_id_list, token_id_list = [], []
    real_chars_list, seq_lens_list = [], []
    for index in range(len(token_list)):
        token_id_list.append(token_list[index][:seq_len])
        entity_id_list.append(entity_list[index][:entity_len])

        real_seq_len = min(len(token_list[index]), seq_len)
        tmp = [1] * real_seq_len
        [tmp.append(0) for _ in range(len(tmp), seq_len)]
        seq_lens_list.append(real_seq_len)
        real_chars_list.append(tmp)

    entity_pad = batcher.batch_sentences(entity_id_list)
    token_pad = batcher.batch_sentences(token_id_list)

    print("The shape of tokens after loading vocab:", token_pad.shape)

    # 按每条数据打包
    features = []
    for index in range(len(token_list)):
        curr_features = [
            entity_pad[index],
            token_pad[index],
            real_chars_list[index],
            seq_lens_list[index],
        ]
        features.append(curr_features)

    return np.array(features)
