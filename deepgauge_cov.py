import numpy as np
import copy
import math

from input_data import load_neural_value


def get_boundary(neuron_value, number_of_neuron):
    """
    对于当前所有样本的神经元输出值neuron_value， 从中找到每个神经元的最大值和最小值
    :param neuron_value: 神经元输出值 -1*number_of_value
    :param number_of_neuron: 神经元个数
    :return:
    """
    max_value = np.max(neuron_value, axis=0)
    min_value = np.min(neuron_value, axis=0)
    boundary = []
    for i in range(number_of_neuron):
        dic = {"max": max_value[i], "min": min_value[i]}
        boundary.append(dic)
        # if min_value[i] == 0:
        #     print(i)
    # print(boundary)
    return boundary


def nbcov_and_snacov(neuron_value, number_of_neuron, boundary):
    """
    计算样本集的神经元边界覆盖率NBCov和强神经元激活覆盖率SNACov
    :param neuron_value:
    :param number_of_neuron:
    :param boundary:
    :return: NBCov, SNACov
    """
    print("开始计算神经元边界覆盖率和强神经元激活覆盖率......")
    count_upper = 0
    count_lower = 0
    for i in range(number_of_neuron):
        upper_flag, lower_flag = 0, 0
        for example in neuron_value:
            if example[i] > boundary[i]["max"] and upper_flag == 0:
                count_upper += 1
                upper_flag = 1
            elif example[i] < boundary[i]["min"] and lower_flag == 0:
                count_lower += 1
                lower_flag = 1
            if upper_flag == 1 and lower_flag == 1:
                break
    # print(count_upper, count_lower)
    print("计算神经元边界覆盖率和强神经元激活覆盖率结束!")
    return (count_upper + count_lower) / (2 * number_of_neuron), count_upper / number_of_neuron


def k_multisection_neuron_coverage(neuron_value, number_of_neuron, boundary, number_of_section):
    """
    得到k多节神经元覆盖率
    :param neuron_value: 对所有样本的神经元激活值 -1*number_of_neuron
    :param number_of_neuron: 神经元个数
    :param boundary: 激活值的边界
    :param number_of_section: 分成的节数量
    :return:
    """
    # 从训练集得到神经元出现过的最大最小值
    print("------开始计算神经元k分覆盖率和强神经元激活覆盖率------")
    k_section_bound = []
    for i in range(number_of_neuron):
        temp = [float(boundary[i]["min"])]
        delta = (boundary[i]["max"] - boundary[i]["min"]) / number_of_section
        #        print(delta)
        k_bound = boundary[i]["min"]
        for _ in range(number_of_section):
            k_bound += delta
            temp.append(k_bound)
        k_section_bound.append(temp)
    # print(k_section_bound)

    count_k_section = 0.0
    for i in range(number_of_neuron):
        flag_k_section = [0 for _ in range(number_of_section)]
        for example in neuron_value:
            # print("example = ", example)
            for k in range(number_of_section):
                if k_section_bound[i][k] < example[i] <= k_section_bound[i][k + 1] and flag_k_section[k] == 0:
                    flag_k_section[k] = 1
                    count_k_section += 1
            if 0 not in flag_k_section:
                break
    # print(count_k_section)
    print("------计算神经元k分覆盖率和强神经元激活覆盖率结束------")
    return count_k_section / (number_of_section * number_of_neuron)


def top_k_neuron_cov(neuron_value, number_of_neuron_layer, k_value):
    """
    计算top-k覆盖率的值
    :param neuron_value: 神经元激活值 -1*number_of_neuron
    :param number_of_neuron_layer: 该层神经元个数
    :param k_value: 前k值
    :return:
    """
    # 标记top_k神经元的索引，大小和神经元个数相同
    top_k_neuron_index = [0 for _ in range(number_of_neuron_layer)]
    # 遍历每个神经元
    for i in range(len(neuron_value)):
        neuron_value_to_list = neuron_value[i].tolist()
        # 得出top-k的下标
        for k in range(k_value):
            # 得到最大值的索引
            max_index = neuron_value_to_list.index(max(neuron_value_to_list))
            neuron_value_to_list[max_index] = -1
            # 如果该索引没被标记过则标记
            if top_k_neuron_index[max_index] == 0:
                top_k_neuron_index[max_index] = 1

    return sum(top_k_neuron_index) / number_of_neuron_layer


if __name__ == "__main__":
    # neuron_value_train = [[0.9, 0.8, 0, 0.1],
    #                       [0.3, 0.9, 0, 0.7],
    #                       [0.2, 0.3, 0.9, 0.3],
    #                       [0.4, 0.2, 0, 0.5]]
    # neuron_value_test = [[0, 0.75, 0, 0.25],
    #                      [1, 0.26, 0, 0.33],
    #                      [0.2, 0, 1, 0.55]]
    #
    # # nb_cov, sna_cov = nbcov_and_snacov(neuron_value_test, 4, get_boundary(neuron_value_train, 4))
    # # k_multi_cov = k_multisection_neuron_coverage(neuron_value_test, 4, get_boundary(neuron_value_train, 4), 2)
    # cov = top_k_neuron_cov(neuron_value_test, 4, 1)
    # print(cov)
    # print("nb_cov = %f, sna_cov = %f, k_multi_cov = %f" % (nb_cov, sna_cov, k_multi_cov))

    neuron_value_list = []
    for j in range(10):
        source_path = r"MNIST_data/testing_data/all_neural_value/layer6_fc_2/class_" + str(j) + "_NeuralValue.txt"
        neuron_number = 84
        temp_neuron_value = load_neural_value(source_path, neuron_number)
        for example_ in temp_neuron_value:
            neuron_value_list.append(example_)
    # print(neuron_value_list)
    # neuron_value_test = [[0, 0.75, 0, 0], [0, 0.26, 0, 0.49], [0.2, 0, 0.76, 0], [0, 0, 0, 0]]

    neuron_value_train = []
    for j in range(10):
        source_path = r"MNIST_data/training_data/all_neural_value/layer6_fc_2/class_" + str(j) + "_NeuralValue.txt"
        neuron_number = 84
        temp_neuron_value = load_neural_value(source_path, neuron_number)
        for example_ in temp_neuron_value:
            neuron_value_train.append(example_)

    # neuron_value_list = []
    # source_path = r"MNIST_data/testing_data/all_neural_value/layer5_fc_1/class_" + str(9) + "_NeuralValue.txt"
    # neuron_number = 120
    # temp_neuron_value = load_neural_value(source_path, neuron_number)
    # for example_ in temp_neuron_value:
    #     neuron_value_list.append(example_)
    #
    # neuron_value_train = []
    # source_path = r"MNIST_data/training_data/all_neural_value/layer5_fc_1/class_" + str(9) + "_NeuralValue.txt"
    # neuron_number = 120
    # temp_neuron_value = load_neural_value(source_path, neuron_number)
    # for example_ in temp_neuron_value:
    #     neuron_value_train.append(example_)

    boundary_result = get_boundary(neuron_value_train, 84)
    cov = nbcov_and_snacov(neuron_value_list, 84, boundary_result)
    # print(boundary_result)
    # cov = top_k_neuron_cov(neuron_value_list, 10*10*16, 3)
    # print(k_multisection_neuron_coverage(neuron_value_list, 28*28*6, boundary_result, 1000))
    # print(k_multisection_neuron_coverage(neuron_value_list, 120, boundary_result, 1000))

    # cov1 = top_k_neuron_cov(neuron_value_list, 120, 1)
    # cov2 = top_k_neuron_cov(neuron_value_list, 120, 2)
    # cov3 = top_k_neuron_cov(neuron_value_list, 120, 3)
    # print(cov1, cov2, cov3)

