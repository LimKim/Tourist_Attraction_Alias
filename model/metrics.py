import sys
import load_data


def calculate_PRF1(tp, fp, fn):
    if tp == 0:
        return 0, 0, 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# 评价指标不是单纯对比标签结果，同样要对识别出来的字符串进行比对
def calculate_metrics(pred_res_list, label_res_list):
    assert len(pred_res_list) == len(label_res_list)
    tp, fp, fn = 0, 0, 0
    for idx in range(len(pred_res_list)):
        pred = [
            str([val[0].split("_")[-1], val[1], val[2]])
            for val in pred_res_list[idx]
            if "False" not in val[0]
        ]
        real = [
            str([val[0].split("_")[-1], val[1], val[2]])
            for val in label_res_list[idx]
            if "False" not in val[0]
        ]

        rec_set = set(pred) | set(real)
        for rec in rec_set:
            if rec in pred and rec in real:
                tp += 1
            elif rec in pred and rec not in real:
                fp += 1
            elif rec not in pred and rec in real:
                fn += 1
    p, r, f1 = calculate_PRF1(tp, fp, fn)
    print("TP:%d\tFP:%d\tFN:%d" % (tp, fp, fn))
    return p, r, f1


def process_labels(label_list, data_file):
    entity_list, token_list, _ = load_data.read_data(data_file)
    label_res_list = []
    assert len(entity_list) == len(token_list) == len(label_list)
    for idx in range(len(label_list)):
        label_res_list.append(get_tag_and_index(label_list[idx], token_list[idx]))
    for idx in range(len(label_res_list)):
        curr_entity = "".join(entity_list[idx])
        label_res_list[idx] = [
            val for val in label_res_list[idx] if "".join(val[-1]) != curr_entity
        ]

    return label_res_list


def further_process_labels(label_list, data_file):
    entity_list, token_list, tag_list = load_data.read_data(data_file)
    for idx in range(len(label_list)):
        manual_rule(entity_list[idx], label_list[idx])
    return label_list


def overlap(word1, word2):
    """
    判断两个词是否存在字符重叠
    """
    for ch in word1:
        if ch in word2:
            return True
    return False


def get_tag_and_index(label_seq, token_seq):
    """
    [识别结果的标签，始末位置，识别出的字符串]，返回该结果的list
    """
    label_rec_seq = []
    start_id = 0
    end_id = 0
    curr_label = ""
    for index, label in enumerate(label_seq):
        if label[0] == "I":
            if curr_label:
                end_id = index + 1
        elif label[0] == "B":
            if curr_label:
                label_rec_seq.append(
                    [curr_label, [start_id, end_id], token_seq[start_id:end_id]]
                )
            start_id = index
            end_id = index + 1
            curr_label = label.split("-")[-1]
        else:
            if curr_label:
                label_rec_seq.append(
                    [curr_label, [start_id, end_id], token_seq[start_id:end_id]]
                )
            curr_label = ""
    if curr_label:
        label_rec_seq.append(
            [curr_label, [start_id, end_id], token_seq[start_id:end_id]]
        )
    return label_rec_seq


def manual_rule(entity, label_rec_seq):
    """
    根据假设后处理识别结果
    1.如果某一相同实体在不同位置被识别为SE和AE，则认为该实体是AE
    2.如果某一实体类别为SE，则其必须与entity具有字符重叠
    3.最终评价标准只看True/False标签
    4.***对于识别结果与原实体一致的情况，不纳入评估***
    """
    multi_labels_str = {}
    for idx in range(len(label_rec_seq)):
        curr_label, _, rec_str = label_rec_seq[idx]
        if str(rec_str) not in multi_labels_str:
            multi_labels_str[str(rec_str)] = set()
        multi_labels_str[str(rec_str)].add(curr_label)

    for rec_str in multi_labels_str:
        if len(multi_labels_str[rec_str]) > 1:
            for idx in range(len(label_rec_seq)):
                if str(label_rec_seq[idx][2]) == rec_str:
                    label_rec_seq[idx][0] = label_rec_seq[idx][0].replace("SE", "AE")

    for idx in range(len(label_rec_seq)):
        if "SE" in label_rec_seq[idx][0] and not overlap(entity, label_rec_seq[idx][2]):
            label_rec_seq[idx][0] = label_rec_seq[idx][0].replace("True", "False")

