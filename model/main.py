import tensorflow as tf
import numpy as np
import json
import argparse
import os
import sys

sys.path.append("../ELMo_self_train")
from model_fn import *
import model_config
import load_data
import metrics


def batch_iter(x, y, is_random=True, batch_size=64):  # 批量生成数据
    # 生成批次数据
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    (x_shuffle, y_shuffle) = (x[indices], y[indices]) if is_random else (x, y)

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def user_options(model_config):
    #   argparse预定义参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--is_train")
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-i", "--model_id", type=int)
    parser.add_argument("-d", "--device", type=int)
    args = parser.parse_args()
    if args.is_train:
        if args.is_train in ["True", "1", "true"]:
            model_config.is_train = True
        elif args.is_train in ["False", "0", "false"]:
            model_config.is_train = False
        else:
            print("Error: unexpected argv 'is_train'")
            exit()
    else:
        print("Error: need specify an action")
        exit()

    if args.batch_size:
        model_config.batch_size = args.batch_size
    if args.model_id:
        model_config.model_id = args.model_id
    if args.device:
        model_config.gpu_id = str(args.device)
    if args.model_id:
        model_config.model_id = args.model_id


def convert_data(x, y):
    """
    将以条为单位的数据 转换为 各自维度的数据矩阵
    """
    entity_pad = np.array([val[0] for val in x])
    token_pad = np.array([val[1] for val in x])
    real_chars_list = np.array([val[2] for val in x])
    seq_lens_list = np.array([val[3] for val in x])
    ner_vec_list = np.array([list(val[0]) for val in y])
    classifier_vec_list = np.array([list(val[1]) for val in y])
    return [
        entity_pad,
        token_pad,
        real_chars_list,
        seq_lens_list,
        ner_vec_list,
        classifier_vec_list,
    ]


def merge_tag(
    ner_label_list,
    classifier_label_list,
    mask_list,
    ner_cate2id_dict,
    classifier_cate2id_dict,
):
    ner_id2cate_dict = {v: k for k, v in ner_cate2id_dict.items()}
    classifier_id2cate_dict = {v: k for k, v in classifier_cate2id_dict.items()}
    assert len(ner_label_list) == len(classifier_label_list) == len(mask_list)
    merge_label_list = []
    for index in range(len(ner_label_list)):
        assert len(ner_label_list[index]) == len(classifier_label_list[index])
        new_label = [
            ner_id2cate_dict[x] + "_" + classifier_id2cate_dict[y]
            if ner_id2cate_dict[x][0] != "O"
            else "O"
            for x, y in zip(ner_label_list[index], classifier_label_list[index])
        ]

        curr_flag = ""
        for idx in range(len(mask_list[index])):
            if mask_list[index][idx] == 0:
                new_label[idx] = "O"
                continue
            if new_label[idx][0] == "B":
                curr_flag = "True" if "False" not in new_label[idx] else "False"
            elif new_label[idx][0] == "I" and curr_flag:
                new_label[idx] = (
                    new_label[idx]
                    .replace("True", curr_flag)
                    .replace("False", curr_flag)
                )
        merge_label_list.append(new_label)
    return merge_label_list


def nn_train(
    model_config, train_data, dev_data, ner_cate2id_dict, classifier_cate2id_dict
):
    if os.path.exists(model_config.save_dir):
        print("SaveVariable dir has exist")
    else:
        os.mkdir(model_config.save_dir)
        print("Making SaveVariable dir")

    train_features, train_labels = train_data
    dev_features, dev_labels = dev_data

    train_nums = len(train_features)
    num_batch = int((train_nums - 1) / model_config.batch_size) + 1  # 总批数

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            visible_device_list=model_config.gpu_id, allow_growth=True
        )
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        )

        # initializer = tf.contrib.layers.xavier_initializer()
        with sess.as_default():
            myModel = globals()[model_config.model_name](model_config)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(
                model_config.lr, global_step, num_batch, 0.98, True
            )
            optimizer = tf.train.RMSPropOptimizer(learning_rate)  # 定义优化器
            train_op = optimizer.minimize(myModel.loss, global_step=global_step)

            print("initializing...")
            print("=" * 40)
            print("Nums of TrainData:", train_nums)
            print("Batch Size:", model_config.batch_size)
            print("Steps per epoch:", train_nums // model_config.batch_size)
            print("=" * 40)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=50)

            def train_action(batch_xs, batch_ys):
                entity_pad, token_pad, real_chars_list, seq_lens_list, ner_vec_list, classifier_vec_list = convert_data(
                    batch_xs, batch_ys
                )

                feed_dict = {
                    myModel.entity_pad: entity_pad,
                    myModel.token_pad: token_pad,
                    myModel.real_chars_list: real_chars_list,
                    myModel.seq_lens_list: seq_lens_list,
                    myModel.ner_vec_list: ner_vec_list,
                    myModel.classifier_vec_list: classifier_vec_list,
                    myModel.keep_prob: 0.5,
                }

                _, step_loss = sess.run((train_op, myModel.loss), feed_dict=feed_dict)
                return step_loss

            epoch_counter = 0
            first_flag = True
            min_loss, max_F1 = float("inf"), 0
            while True:
                epoch_counter += 1

                train_batch = batch_iter(
                    train_features, train_labels, batch_size=model_config.batch_size
                )
                for batch_xs, batch_ys in train_batch:  # 一个循环作为一个epoch
                    step_loss = train_action(batch_xs, batch_ys)
                    current_step = tf.train.global_step(sess, global_step)  # step + 1

                    if current_step % 50 == 0:  # 每50步显示当前batch的训练集上的实验结果
                        print("steps: %-7d\tloss:%2.4f" % (current_step, step_loss))

                    if current_step % model_config.dev_per_steps == 0:
                        print("=" * 40)
                        print("evaluating in dev...")
                        dev_batch = batch_iter(
                            dev_features, dev_labels, is_random=False, batch_size=20
                        )

                        loss_sum = 0
                        total_nums = 0
                        ner_pred_list, classifier_pred_list = [], []
                        ner_label_list, classifier_label_list = [], []
                        mask_list = []
                        for one_batch in dev_batch:
                            batch_xs, batch_ys = one_batch
                            current_batch = len(batch_xs)
                            entity_pad, token_pad, real_chars_list, seq_lens_list, ner_vec_list, classifier_vec_list = convert_data(
                                batch_xs, batch_ys
                            )
                            feed_dict = {
                                myModel.entity_pad: entity_pad,
                                myModel.token_pad: token_pad,
                                myModel.real_chars_list: real_chars_list,
                                myModel.seq_lens_list: seq_lens_list,
                                myModel.ner_vec_list: ner_vec_list,
                                myModel.classifier_vec_list: classifier_vec_list,
                                myModel.keep_prob: 1,
                            }

                            batch_loss, batch_ner_pred, batch_ner_label, batch_classifier_pred, batch_classifier_label = sess.run(
                                (
                                    myModel.loss,
                                    myModel.decode_tags,
                                    myModel.ner_labels,
                                    myModel.classifier_predictions,
                                    myModel.classifier_labels,
                                ),
                                feed_dict=feed_dict,
                            )

                            loss_sum += batch_loss * current_batch
                            total_nums += current_batch
                            ner_pred_list.append(batch_ner_pred)
                            ner_label_list.append(batch_ner_label)
                            classifier_pred_list.append(batch_classifier_pred)
                            classifier_label_list.append(batch_classifier_label)
                            mask_list.append(real_chars_list)

                        ner_pred_list = np.concatenate(ner_pred_list, 0).tolist()
                        ner_label_list = np.concatenate(ner_label_list, 0).tolist()
                        mask_list = np.concatenate(mask_list, 0).tolist()
                        classifier_pred_list = np.concatenate(
                            classifier_pred_list, 0
                        ).tolist()
                        classifier_label_list = np.concatenate(
                            classifier_label_list, 0
                        ).tolist()

                        prediction_list = merge_tag(
                            ner_pred_list,
                            classifier_pred_list,
                            mask_list,
                            ner_cate2id_dict,
                            classifier_cate2id_dict,
                        )
                        label_list = merge_tag(
                            ner_label_list,
                            classifier_label_list,
                            mask_list,
                            ner_cate2id_dict,
                            classifier_cate2id_dict,
                        )

                        loss_ = loss_sum / total_nums
                        prediction_list_pro = metrics.process_labels(
                            prediction_list, model_config.dev_path
                        )
                        label_list_pro = metrics.process_labels(
                            label_list, model_config.dev_path
                        )
                        precision, recall, F1 = metrics.calculate_metrics(
                            prediction_list_pro, label_list_pro
                        )
                        print("step:%-7d\tloss:%2.4f" % (current_step, loss_))
                        print(
                            "Precision:%2.4f  Recall:%2.4f  F1:%2.4f"
                            % (precision, recall, F1)
                        )
                        print("After manual processed..")
                        prediction_list_pro = metrics.further_process_labels(
                            prediction_list_pro, model_config.dev_path
                        )
                        precision, recall, F1 = metrics.calculate_metrics(
                            prediction_list_pro, label_list_pro
                        )
                        print(
                            "Precision:%2.4f  Recall:%2.4f  F1:%2.4f"
                            % (precision, recall, F1)
                        )

                        if loss_ < min_loss or F1 > max_F1:  # 保存模型
                            min_loss = min(loss_, min_loss)
                            max_F1 = max(F1, max_F1)

                            save_path = saver.save(
                                sess,
                                model_config.models_dir + str(current_step) + ".ckpt",
                            )
                            print("save to path:", save_path)
                            if first_flag:
                                log_path = model_config.models_dir + "best_model_id.log"
                                if os.path.exists(log_path):
                                    os.remove(log_path)
                                fw = open(log_path, "w", encoding="utf8")
                                first_flag = False
                            else:
                                fw = open(
                                    model_config.models_dir + "best_model_id.log",
                                    "a",
                                    encoding="utf8",
                                )
                            fw.write(
                                "step:%-7d\tloss:%2.4f\tPrecision:%2.4f  Recall:%2.4f  F1:%2.4f\n"
                                % (current_step, loss_, precision, recall, F1)
                            )
                            fw.close()
                        print("=" * 40)


def nn_test(model_config, test_data, ner_cate2id_dict, classifier_cate2id_dict):
    id2cate_dict = {v: k for k, v in classifier_cate2id_dict.items()}
    print(id2cate_dict)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            visible_device_list=model_config.gpu_id, allow_growth=True
        )
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        )
        # initializer = tf.contrib.layers.xavier_initializer()
        myModel = globals()[model_config.model_name](model_config)
        print("initializing...")
        sess.run(tf.global_variables_initializer())

        print("loading model by id:", model_config.model_id)
        saver = tf.train.Saver()
        saver.restore(
            sess, model_config.models_dir + str(model_config.model_id) + ".ckpt"
        )
        test_features, test_labels = test_data
        # for val in test_labels:
        #     print(val)
        test_batch = batch_iter(
            test_features, test_labels, is_random=False, batch_size=20
        )

        ner_pred_list, classifier_pred_list = [], []
        ner_label_list, classifier_label_list = [], []
        mask_list = []
        for batch_xs, batch_ys in test_batch:
            entity_pad, token_pad, real_chars_list, seq_lens_list, ner_vec_list, classifier_vec_list = convert_data(
                batch_xs, batch_ys
            )
            feed_dict = {
                myModel.entity_pad: entity_pad,
                myModel.token_pad: token_pad,
                myModel.real_chars_list: real_chars_list,
                myModel.seq_lens_list: seq_lens_list,
                myModel.ner_vec_list: ner_vec_list,
                myModel.classifier_vec_list: classifier_vec_list,
                myModel.keep_prob: 1,
            }
            batch_ner_pred, batch_ner_label, batch_classifier_pred, batch_classifier_label = sess.run(
                (
                    myModel.decode_tags,
                    myModel.ner_labels,
                    myModel.classifier_predictions,
                    myModel.classifier_labels,
                ),
                feed_dict=feed_dict,
            )
            ner_pred_list.append(batch_ner_pred)
            ner_label_list.append(batch_ner_label)
            classifier_pred_list.append(batch_classifier_pred)
            classifier_label_list.append(batch_classifier_label)
            mask_list.append(real_chars_list)

        ner_pred_list = np.concatenate(ner_pred_list, 0).tolist()
        ner_label_list = np.concatenate(ner_label_list, 0).tolist()
        classifier_pred_list = np.concatenate(classifier_pred_list, 0).tolist()
        classifier_label_list = np.concatenate(classifier_label_list, 0).tolist()
        mask_list = np.concatenate(mask_list, 0).tolist()

        prediction_list = merge_tag(
            ner_pred_list,
            classifier_pred_list,
            mask_list,
            ner_cate2id_dict,
            classifier_cate2id_dict,
        )
        label_list = merge_tag(
            ner_label_list,
            classifier_label_list,
            mask_list,
            ner_cate2id_dict,
            classifier_cate2id_dict,
        )

        prediction_list_pro = metrics.process_labels(
            prediction_list, model_config.test_path
        )
        label_list_pro = metrics.process_labels(label_list, model_config.test_path)

        precision, recall, F1 = metrics.calculate_metrics(
            prediction_list_pro, label_list_pro
        )
        print("Precision:%2.4f  Recall:%2.4f  F1:%2.4f" % (precision, recall, F1))

        print("After manual processed..")
        prediction_list_pro = metrics.further_process_labels(
            prediction_list_pro, model_config.test_path
        )
        precision, recall, F1 = metrics.calculate_metrics(
            prediction_list_pro, label_list_pro
        )
        print("Precision:%2.4f  Recall:%2.4f  F1:%2.4f" % (precision, recall, F1))

        fw = open("test.tag", "w", encoding="utf8")
        for ssi in prediction_list:
            fw.write(str(ssi) + "\n")

        fw = open("label.tag", "w", encoding="utf8")
        for ssi in label_list:
            fw.write(str(ssi) + "\n")


if __name__ == "__main__":
    curr_config = model_config.config_option()
    user_options(curr_config)

    word_dict, ner_info, classifier_info = load_data.get_vocab(curr_config.train_path)
    ner_category, ner_cate2id_dict = ner_info
    classifier_category, classifier_cate2id_dict = classifier_info

    curr_config.vocab_size = len(word_dict)
    curr_config.ner_size = len(ner_cate2id_dict)
    curr_config.classifier_size = len(classifier_cate2id_dict)
    print("=" * 40)
    for k, v in ner_cate2id_dict.items():
        print("%-10s\t%d" % (k, v))
    print("=" * 40)
    for k, v in classifier_cate2id_dict.items():
        print("%-10s\t%d" % (k, v))
    print("=" * 40)
    print("NER_size", curr_config.ner_size)
    print("Classifier_size", curr_config.classifier_size)

    #   保存变量
    if curr_config.is_train:
        train_features = load_data.load_ELMo_data(
            curr_config.train_path, curr_config.seq_len, curr_config.entity_len
        )
        train_labels = load_data.load_tag_vec(
            curr_config.train_path,
            ner_cate2id_dict,
            classifier_cate2id_dict,
            curr_config.seq_len,
        )

        dev_features = load_data.load_ELMo_data(
            curr_config.dev_path, curr_config.seq_len, curr_config.entity_len
        )
        dev_labels = load_data.load_tag_vec(
            curr_config.dev_path,
            ner_cate2id_dict,
            classifier_cate2id_dict,
            curr_config.seq_len,
        )
        nn_train(
            curr_config,
            [train_features, train_labels],
            [dev_features, dev_labels],
            ner_cate2id_dict,
            classifier_cate2id_dict,
        )
    else:
        test_features = load_data.load_ELMo_data(
            curr_config.test_path, curr_config.seq_len, curr_config.entity_len
        )
        test_labels = load_data.load_tag_vec(
            curr_config.test_path,
            ner_cate2id_dict,
            classifier_cate2id_dict,
            curr_config.seq_len,
        )
        nn_test(
            curr_config,
            [test_features, test_labels],
            ner_cate2id_dict,
            classifier_cate2id_dict,
        )
