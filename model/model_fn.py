import tensorflow as tf
import numpy as np
import h5py
import sys

sys.path.append("../ELMo_self_train")
from bilm import (
    TokenBatcher,
    BidirectionalLanguageModel,
    weight_layers,
    dump_token_embeddings,
)


class JointModel:
    def __init__(self, nn_config):
        with tf.name_scope("input_scope"):
            # 载入ELMo向量，序列左右各增加一个长度
            self.token_pad = tf.placeholder(
                tf.int32, [None, nn_config.seq_len + 2], name="token_pad"
            )  # 序列输入
            self.entity_pad = tf.placeholder(
                tf.int32, [None, nn_config.seq_len + 2], name="entity_pad"
            )  # 主体输入
            self.ner_vec_list = tf.placeholder(
                tf.int32,
                [None, nn_config.seq_len, nn_config.ner_size],
                name="ner_vec_list",
            )  # ner标签输入
            self.classifier_vec_list = tf.placeholder(
                tf.int32,
                [None, nn_config.seq_len, nn_config.classifier_size],
                name="classifier_vec_list",
            )  # 分类标签输入
            self.real_chars_list = tf.placeholder(
                tf.int32, [None, nn_config.seq_len], name="real_chars_list"
            )  # 01标签，控制序列实际输入大小，padding为0，否则为1
            self.seq_lens_list = tf.placeholder(
                tf.int32, [None], name="seq_lens_list"
            )  # 每个序列的实际长度

            self.keep_prob = tf.placeholder(tf.float32)
            self.alpha = tf.Variable(tf.zeros([1]), dtype=tf.float32)
            current_batch = tf.shape(self.token_pad)[0]

            print("=" * 40)
            print("The shape of token_pad:", self.token_pad.shape)
            print("The shape of entity_pad:", self.entity_pad.shape)
            print("The shape of ner_vec_list:", self.ner_vec_list.shape)
            print("The shape of classifier_vec_list:", self.classifier_vec_list.shape)
            print("The shape of real_chars_list:", self.real_chars_list.shape)
            print("The shape of seq_lens_list:", self.seq_lens_list.shape)
            print("=" * 40)

        with tf.name_scope("load_ELMo_scope"):
            print("Using pretrained ELMo word embedding...")

            prefix = "./ELMo_file"  # 相对路径
            vocab_file = prefix + "/vocab.txt"  # 词汇表
            options_file = prefix + "/options.json"
            weight_file = prefix + "/weights.hdf5"
            token_embedding_file = prefix + "/laiye_vocab_embedding.hdf5"

            # Build the biLM graph.
            bilm = BidirectionalLanguageModel(
                options_file,
                weight_file,
                use_character_inputs=False,
                embedding_weight_file=token_embedding_file,
            )

            text_embeddings_op = bilm(self.token_pad)
            entity_embeddings_op = bilm(self.entity_pad)

            #   数据
            print("Embedding ELMo input...")
            # 相当于 tf.nn.embedding_lookup()
            elmo_text_input = weight_layers("tokens", text_embeddings_op, l2_coef=0.0)

            elmo_entity_input = weight_layers(
                "entity", entity_embeddings_op, l2_coef=0.0
            )
            self.embedded_chars = elmo_text_input["weighted_op"]
            self.entity_embedding = elmo_entity_input["weighted_op"]
            self.entity_embedding = self.entity_embedding[:, : nn_config.entity_len]
            nn_config.word_dim = 600
            # 输入
            input_forward = self.embedded_chars

            print("=" * 40)
            print(
                "The shape of word embedding after loading ELMo：",
                self.embedded_chars.shape,
            )
            print(
                "The shape of entity embedding after loading ELMo：",
                self.entity_embedding.shape,
            )
            print("=" * 40)

        with tf.variable_scope("ner_model"):
            bilstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=nn_config.num_units)
            bilstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=nn_config.num_units)

            fw_h0 = bilstm_fw_cell.zero_state(current_batch, dtype=tf.float32)
            bw_h0 = bilstm_bw_cell.zero_state(current_batch, dtype=tf.float32)

            # 双向lstm，outputs[0]为正向的，[1]为反向，state同理，且outputs[0]取[-1]，outputs[1]取[0]
            (outputs_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                bilstm_fw_cell,
                bilstm_bw_cell,
                input_forward,
                initial_state_fw=fw_h0,
                initial_state_bw=bw_h0,
            )

            self.hidden_output = tf.concat([outputs_fw, output_bw], 2)

            ner_linear_output = tf.reshape(
                self.hidden_output, shape=[-1, nn_config.num_units * 2]
            )
            ner_logits = tf.layers.dense(ner_linear_output, nn_config.ner_size)

            # 未经过CRF层的ner预测标签得分
            ner_logits = tf.layers.dropout(
                tf.reshape(
                    ner_logits, shape=[-1, nn_config.seq_len, nn_config.ner_size]
                ),
                rate=self.keep_prob,
            )

            self.ner_labels = tf.reshape(
                tf.argmax(self.ner_vec_list, axis=-1), shape=[-1, nn_config.seq_len]
            )
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                ner_logits, self.ner_labels, self.seq_lens_list
            )
            ner_loss = tf.reduce_mean(-log_likelihood)

            # 最终的ner预测标签结果
            self.decode_tags, best_score = tf.contrib.crf.crf_decode(
                ner_logits, transition_params, self.seq_lens_list
            )

        with tf.variable_scope("classifier_model"):
            # 将序列长度扩展entity_len-1个长度
            sent_emb = tf.concat(
                [
                    self.embedded_chars,
                    tf.zeros(
                        [current_batch, nn_config.entity_len - 1, nn_config.word_dim]
                    ),
                ],
                axis=1,
            )

            # 随机初始化卷积核和偏置
            # 窗口大小为entity_len
            sent_weights = tf.Variable(
                tf.truncated_normal(
                    [nn_config.entity_len, nn_config.word_dim, 1, nn_config.num_units],
                    stddev=0.1,
                )
            )
            entity_weights = tf.Variable(
                tf.truncated_normal(
                    [nn_config.entity_len, nn_config.word_dim, 1, nn_config.num_units],
                    stddev=0.1,
                )
            )
            sent_bias = tf.Variable(tf.zeros([nn_config.num_units]))
            entity_bias = tf.Variable(tf.zeros([nn_config.num_units]))

            sent_emb = tf.reshape(
                sent_emb,
                [
                    -1,
                    nn_config.seq_len + nn_config.entity_len - 1,
                    nn_config.word_dim,
                    1,
                ],
            )
            entity_emb = tf.reshape(
                self.entity_embedding, [-1, nn_config.entity_len, nn_config.word_dim, 1]
            )
            sent_conv = tf.nn.conv2d(
                sent_emb, sent_weights, strides=[1, 1, 1, 1], padding="VALID"
            )
            entity_conv = tf.nn.conv2d(
                entity_emb, entity_weights, strides=[1, 1, 1, 1], padding="VALID"
            )

            # 模拟窗口滑动卷积操作
            sent_h = tf.nn.relu(
                tf.nn.bias_add(sent_conv, sent_bias)
                + tf.nn.bias_add(entity_conv, entity_bias)
            )  # [-1, 100, 1, num_units]

            sent_h = tf.reshape(sent_h, [-1, nn_config.seq_len, nn_config.num_units])

            # 将每个时刻与主体卷积的结果与ner的hidden_output拼接作分类
            classifier_linear_output = tf.reshape(
                tf.concat([self.hidden_output, sent_h], -1),
                [-1, nn_config.num_units * 3],
            )
            classifier_logits = tf.layers.dense(
                classifier_linear_output, nn_config.classifier_size
            )
            classifier_logits = tf.reshape(
                classifier_logits,
                shape=[-1, nn_config.seq_len, nn_config.classifier_size],
            )
            self.classifier_labels = tf.argmax(self.classifier_vec_list, -1)
            self.classifier_predictions = tf.argmax(classifier_logits, -1)

        with tf.variable_scope("mask_layer"):

            # mask层维度等于ner预测标签的维度
            # mask层对应位置的预测标签为B时，值为1，否则为0
            # 手动设置b_index的值，由于B-AE在ner标签的one_hot中index为0，B-SE为1
            # 则令b_index=1，ner_labels-b_index后，B-AE结果为-1，B-SE结果为0，其他标签结果为正数
            # 先取符号结果sign，再-1，得到B-AE结果为-2，B-SE结果为-1，其他标签结果为0
            # 再对结果取绝对值abs，B-AE结果为2，B-SE结果为1，其他标签结果为0
            # 最后再取符号结果，B标签结果为1，其他标签结果为0，即最终mask层的01结果
            b_index = 1
            mask_matrix = tf.sign(
                tf.abs(tf.sign(self.classifier_predictions - b_index) - 1)
            )
            mask_matrix = tf.cast(mask_matrix, tf.float32)

            # keep_matrix表示最终纳入loss统计的位置，1表示保留该位置loss，0表示忽略该位置loss
            real_chars = tf.cast(self.real_chars_list, tf.float32)
            keep_matrix = tf.multiply(real_chars, mask_matrix)

        with tf.variable_scope("loss"):
            classifier_loss_list = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.classifier_vec_list, logits=classifier_logits
            )
            classifier_loss_list = tf.reshape(
                classifier_loss_list, [-1, nn_config.seq_len]
            )
            classifier_loss = tf.reduce_mean(
                tf.multiply(keep_matrix, classifier_loss_list)
            )

            # self.loss = tf.sigmoid(self.alpha)*ner_loss + (1-tf.sigmoid(self.alpha))*classifier_loss
            self.loss = ner_loss + classifier_loss
