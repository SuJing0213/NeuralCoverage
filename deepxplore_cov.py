from input_data import load_neural_value, get_average_conv_pool
import numpy as np

# neuron_coverage in DeepXplore
def neuron_coverage(neuron_value, neuron_num, activate_bound):
    """
    以一定的激活阈值，计算样本集合的神经元激活覆盖率
    :param neuron_value: 所有样本的神经元输出值 -1*number_of_neural
    :param neuron_num: 网络中的神经元个数
    :param activate_bound: 激活阈值
    :return:
    """
    # print("开始计算神经元覆盖率...")
    activated_num = 0.0
    for i in range(neuron_num):
        for example in neuron_value:
            # print(type(example))
            if example[i] > activate_bound:
                activated_num += 1
                break
    # print("神经元覆盖率计算结束!")
    return activated_num / neuron_num


def calculate_cumulative_ncov_rate(neuron_value, neuron_num, activate_bound):
    """

    :param neuron_value:
    :param neuron_num:
    :param activate_bound:
    :return:
    """
    # cum_rate = []
    # for i in range(len(neuron_value)):
    #     rate = neuron_coverage(neuron_value[:i], neuron_num, activate_bound)
    #     print(i, ":", rate)
    #     cum_rate.append(rate)
    # return cum_rate

    cum_rate = [0]
    cov_ed = [0 for _ in range(neuron_num)]
    for instance in neuron_value:
        for index in range(neuron_num):
            if instance[index] > activate_bound:
                cov_ed[index] = 1
        cum_rate.append(sum(cov_ed) / neuron_num)
    return cum_rate


if __name__ == '__main__':

    # -------------------------------------lay1_conv_1-----------------------------------------------------
    # neuron_value_list_temp = []
    # neuron_number = 28 * 28 * 6
    # for i in range(10):
    #     print("加载数据到第%d类" % i)
    #     source_path = r"MNIST_data/training_data/all_neural_value/layer1_conv_1/class_"
    #                   + str(i) + "_NeuralValue.txt"
    #     temp_neuron_value = load_neural_value(source_path, neuron_number)
    #     temp_neuron_value = np.array(temp_neuron_value).reshape([len(temp_neuron_value) * neuron_number])
    #     neuron_value_list_temp += list(temp_neuron_value)
    # neuron_value_list = get_average_conv_pool(neuron_value_list_temp, [-1, 28, 28, 6])
    # act_bound = 0
    # cov = neuron_coverage(neuron_value_list, 6, act_bound)
    # print("active_bound = %f cov = %f" % (act_bound, cov))

    # --------------------------layer2_maxpool_2------------------------------------------------------
    # neuron_value_list_temp = []
    # neuron_number = 14 * 14 * 6
    # for i in range(10):
    #     print("加载数据到第%d类" % i)
    #     source_path = r"MNIST_data/training_data/all_neural_value/layer2_maxpool_1/class_"\
    #                   + str(i) + "_NeuralValue.txt"
    #     temp_neuron_value = load_neural_value(source_path, neuron_number)
    #     temp_neuron_value = np.array(temp_neuron_value).reshape([len(temp_neuron_value) * neuron_number])
    #     neuron_value_list_temp += list(temp_neuron_value)
    # neuron_value_list = get_average_conv_pool(neuron_value_list_temp, [-1, 14, 14, 6])
    # act_bound = 0
    # cov = neuron_coverage(neuron_value_list, 6, act_bound)
    # print("active_bound = %f cov = %f" % (act_bound, cov))

    # --------------------------layer3_conv_1------------------------------------------------------
    # neuron_value_list_temp = []
    # neuron_number = 14 * 14 * 16
    # for i in range(10):
    #     print("加载数据到第%d类" % i)
    #     source_path = r"MNIST_data/training_data/all_neural_value/layer3_conv_2/class_" \
    #                   + str(i) + "_NeuralValue.txt"
    #     temp_neuron_value = load_neural_value(source_path, neuron_number)
    #     temp_neuron_value = np.array(temp_neuron_value).reshape([len(temp_neuron_value) * neuron_number])
    #     neuron_value_list_temp += list(temp_neuron_value)
    # neuron_value_list = get_average_conv_pool(neuron_value_list_temp, [-1, 14, 14, 16])
    # act_bound = 0
    # cov = neuron_coverage(neuron_value_list, 6, act_bound)
    # print("active_bound = %f cov = %f" % (act_bound, cov))

    # --------------------------layer4_maxpool_2------------------------------------------------------
    # neuron_value_list_temp = []
    # neuron_number = 7 * 7 * 16
    # for i in range(10):
    #     print("加载数据到第%d类" % i)
    #     source_path = r"MNIST_data/training_data/all_neural_value/layer4_maxpool_2/class_"
    #                   + str(i) + "_NeuralValue.txt"
    #     temp_neuron_value = load_neural_value(source_path, neuron_number)
    #     temp_neuron_value = np.array(temp_neuron_value).reshape([len(temp_neuron_value) * neuron_number])
    #     neuron_value_list_temp += list(temp_neuron_value)
    # neuron_value_list = get_average_conv_pool(neuron_value_list_temp, [-1, 7, 7, 16])
    # act_bound = 0
    # cov = neuron_coverage(neuron_value_list, 16, act_bound)
    # print("active_bound = %f cov = %f" % (act_bound, cov))

    # --------------------------layer5_fc_1------------------------------------------------------
    # neuron_value_list = []
    # for i in range(10):
    #     source_path = r"MNIST_data/training_data/all_neural_value/layer5_fc_1/class_" + str(i) + "_NeuralValue.txt"
    #     neuron_number = 120
    #     temp_neuron_value = load_neural_value(source_path, neuron_number)
    #     for example in temp_neuron_value:
    #         neuron_value_list.append(example)
    # act_bound = 0
    # cov = neuron_coverage(neuron_value_list, 120, act_bound)
    # print("active_bound = %f cov = %f" % (act_bound, cov))

    # --------------------------layer6_fc_2------------------------------------------------------
    # neuron_value_list = []
    # for i in range(10):
    #     source_path = r"MNIST_data/training_data/all_neural_value/layer6_fc_2/class_" + str(i) + "_NeuralValue.txt"
    #     neuron_number = 84
    #     temp_neuron_value = load_neural_value(source_path, neuron_number)
    #     for example in temp_neuron_value:
    #         neuron_value_list.append(example)
    # act_bound = 0
    # cov = neuron_coverage(neuron_value_list, 84, act_bound)
    # print("active_bound = %f cov = %f" % (act_bound, cov))

    neuron_value_list_temp = []
    neuron_number = 28 * 28 * 6
    for i in range(10):
        print("加载数据到第%d类" % i)
        source_path = r"MNIST_data/training_data/all_neural_value/layer1_conv_1/class_"\
                      + str(i) + "_NeuralValue.txt"
        temp_neuron_value = load_neural_value(source_path, neuron_number)
        temp_neuron_value = np.array(temp_neuron_value).reshape([len(temp_neuron_value) * neuron_number])
        neuron_value_list_temp += list(temp_neuron_value)
    neuron_value_list_layer1 = get_average_conv_pool(neuron_value_list_temp, [-1, 28, 28, 6])

    neuron_value_list_temp = []
    neuron_number = 14 * 14 * 6
    for i in range(10):
        print("加载数据到第%d类" % i)
        source_path = r"MNIST_data/training_data/all_neural_value/layer2_maxpool_1/class_"\
                      + str(i) + "_NeuralValue.txt"
        temp_neuron_value = load_neural_value(source_path, neuron_number)
        temp_neuron_value = np.array(temp_neuron_value).reshape([len(temp_neuron_value) * neuron_number])
        neuron_value_list_temp += list(temp_neuron_value)
    neuron_value_list_layer2 = get_average_conv_pool(neuron_value_list_temp, [-1, 14, 14, 6])

    neuron_value_list_temp = []
    neuron_number = 14 * 14 * 16
    for i in range(10):
        print("加载数据到第%d类" % i)
        source_path = r"MNIST_data/training_data/all_neural_value/layer3_conv_2/class_" \
                      + str(i) + "_NeuralValue.txt"
        temp_neuron_value = load_neural_value(source_path, neuron_number)
        temp_neuron_value = np.array(temp_neuron_value).reshape([len(temp_neuron_value) * neuron_number])
        neuron_value_list_temp += list(temp_neuron_value)
    neuron_value_list_layer3 = get_average_conv_pool(neuron_value_list_temp, [-1, 14, 14, 16])

    neuron_value_list_temp = []
    neuron_number = 7 * 7 * 16
    for i in range(10):
        print("加载数据到第%d类" % i)
        source_path = r"MNIST_data/training_data/all_neural_value/layer4_maxpool_2/class_"\
                      + str(i) + "_NeuralValue.txt"
        temp_neuron_value = load_neural_value(source_path, neuron_number)
        temp_neuron_value = np.array(temp_neuron_value).reshape([len(temp_neuron_value) * neuron_number])
        neuron_value_list_temp += list(temp_neuron_value)
    neuron_value_list_layer4 = get_average_conv_pool(neuron_value_list_temp, [-1, 7, 7, 16])

    neuron_value_list_layer5 = []
    for i in range(10):
        source_path = r"MNIST_data/training_data/all_neural_value/layer5_fc_1/class_" + str(i) + "_NeuralValue.txt"
        neuron_number = 120
        temp_neuron_value = load_neural_value(source_path, neuron_number)
        for example in temp_neuron_value:
            neuron_value_list_layer5.append(example)

    neuron_value_list_layer6 = []
    for i in range(10):
        source_path = r"MNIST_data/training_data/all_neural_value/layer6_fc_2/class_" + str(i) + "_NeuralValue.txt"
        neuron_number = 84
        temp_neuron_value = load_neural_value(source_path, neuron_number)
        for example in temp_neuron_value:
            neuron_value_list_layer6.append(example)

    global_neuron_value_list = []
    global_neuron_number = 6+6+16+16+120+84

    for i in range(len(neuron_value_list_layer6)):
        global_neuron_value_list.append(list(neuron_value_list_layer1[i]) + list(neuron_value_list_layer2[i])
                                        + list(neuron_value_list_layer3[i]) + list(neuron_value_list_layer4[i])
                                        + list(neuron_value_list_layer5[i]) + list(neuron_value_list_layer6[i]))
    act_bound = 0
    cov = neuron_coverage(global_neuron_value_list, global_neuron_number, act_bound)
    print("active_bound = %f cov = %f" % (act_bound, cov))

