import os


class config_option:
    # gpu_options = tf.GPUOptions(visible_device_list=0,allow_growth=True)
    load_data_dir = ""
    train_path = "ModelData/train.txt"
    dev_path = "ModelData/dev.txt"
    test_path = "ModelData/test.txt"

    save_dir = "saveVariable/"
    models_dir = save_dir + "JointModel/"
    model_name = "JointModel"
    # model_name = "BiLSTM"
    model_id = ""

    #   模型参数
    seq_len = 100  # 句子前100个词
    entity_len = 10  # 主体最大长度限定为10
    vocab_size = 0  # 不同词的个数
    ner_size = 5  # ner标签数
    classifier_size = 2  # classifier标签数
    word_dim = 600  # 词向量的维度
    batch_size = 8
    num_units = 512  # 隐含层单元个数
    is_train = True
    lr = 1e-3
    dev_per_steps = 500
    gpu_id = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存
    # config.gpu_options.allow_growth = True

    # def name_model_name(self):
    #     self.models_dir = (
    #         self.save_dir + self.model_name + "-" + self.model_version + "/"
    #         if self.model_version
    #         else self.save_dir + self.model_name + "/"
    #     )

