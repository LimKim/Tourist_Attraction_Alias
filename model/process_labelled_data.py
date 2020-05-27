# -*- coding:utf-8 -*-
import sys
import re
import os

pinyin = "āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜ"
sign_list = ["'", "’", "-", ".", "?"]


def process_single_sentence(line):
    """
    对单句进行标签抽取
    """
    line = line.strip()
    entity, txt = line.split("\t\t")
    tag_seq = []
    match_res = re.search("\[[^\[]*?((AE)|(SE))[^\[]*?\]", txt)
    while match_res:
        start = match_res.start()
        match_item = match_res.group()
        word_tag = re.search("/((AE)|(SE)).*?\]", match_item).group()
        att = match_item.lstrip("[").rstrip(word_tag)
        word_tag = word_tag.lstrip("/").rstrip("]")
        for _ in range(len(tag_seq), start):
            tag_seq.append("O")
        tag_seq += ["B-" + word_tag] + ["I-" + word_tag] * (len(att) - 1)
        txt = txt.replace(match_item, att, 1)
        match_res = re.search("\[[^\[]*?((AE)|(SE))[^\[]*?\]", txt)

    for _ in range(len(tag_seq), len(txt)):
        tag_seq.append("O")

    new_txt, new_tag_seq = merge_token(txt, tag_seq)
    return [entity, new_txt, new_tag_seq]


def merge_token(ll, tag_seq):
    """
    将英文单词合并为一个token，以免过多占用序列长度
    """
    new_txt, new_tag_seq = [], []
    words = ""
    last_tag = ""
    for index, ch in enumerate(ll):
        if ch.encode("UTF-8").isalpha() or ch in list(pinyin) + sign_list:
            if words and last_tag[1:] == tag_seq[index][1:]:
                words += ch
            else:
                if words:
                    new_txt.append(words)
                    new_tag_seq.append(last_tag)
                words = ch
                last_tag = tag_seq[index]
        else:
            if words:
                new_txt.append(words)
                new_tag_seq.append(last_tag)
                words = ""
            new_txt.append(ch)
            new_tag_seq.append(tag_seq[index])

    if len(new_txt) != len(new_tag_seq):
        raise IndexError
    return new_txt, new_tag_seq


def split_data(dataset):
    """
    将标注数据按8：1：1，切分为训练集(train)、验证集(dev)、测试集(test)
    """
    length = len(dataset)
    train = dataset[: int(length * 0.8)]
    dev = dataset[int(length * 0.8) : int(length * 0.9)]
    test = dataset[int(length * 0.9) :]
    print("Split rate: %d,%d" %(length*0.8, length*0.9))
    return train, dev, test


def write2file(targetfile, data):
    fw = open(targetfile, "w", encoding="utf8")
    for lst in data:
        entity, txt, tag_seq = lst
        fw.write(entity + "\t" + "<head>")
        fw.write("\n")
        merged = zip(txt, tag_seq)
        for token, tag in merged:
            fw.write(token + "\t" + tag)
            fw.write("\n")
        fw.write("\n")


def file_based_process(filename):
    """
    将标注文件转化为模型方便读取格式的训练集验证集测试集
    """
    fr = open(filename, "r", encoding="utf8")
    total_data = []
    for line in fr:
        res_list = process_single_sentence(line)
        total_data.append(res_list)
    train, dev, test = split_data(total_data)
    input_dir = "ModelData"
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    write2file(input_dir + "/train.txt", train)
    write2file(input_dir + "/dev.txt", dev)
    write2file(input_dir + "/test.txt", test)


if __name__ == "__main__":
    prefix = "../Corpus"
    file_based_process(prefix + "/labeled_data.5k.txt")
