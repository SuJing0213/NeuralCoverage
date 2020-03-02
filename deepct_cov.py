import math
from input_data import load_neural_value


def two_way_sparse_cov(neuron_value, number_of_neuron_layer, boundary):
    """
    1 把得到的神经元结点值写成激活/不激活
    2 神经元结点组合成2路
    3 对每个组合看是否被覆盖
    4 计算覆盖率

    :param boundary:
    :param neuron_value:
    :param number_of_neuron_layer:
    :return:

    """
    # 神经元结点是否激活
    neuron_value_activate = neuron_value[:]
    bi_neuron_value_activate = [[0 for _ in range(number_of_neuron_layer)] for __ in range(len(neuron_value))]
    for i in range(len(neuron_value_activate)):
        for j in range(number_of_neuron_layer):
            if neuron_value_activate[i][j] > boundary:
                bi_neuron_value_activate[i][j] = 1
            else:
                bi_neuron_value_activate[i][j] = 0

    # 神经元结点值的组合
    count_combine_activate = 0
    count_combine = 0
    for i in range(number_of_neuron_layer):
        for j in range(i + 1, number_of_neuron_layer):
            count_combine += 1
            flag_combine = [0, 0, 0, 0]
            for example in bi_neuron_value_activate:
                if example[i] == 0 and example[j] == 0:
                    flag_combine[0] = 1
                elif example[i] == 0 and example[j] == 1:
                    flag_combine[1] = 1
                elif example[i] == 1 and example[j] == 0:
                    flag_combine[2] = 1
                elif example[i] == 1 and example[j] == 1:
                    flag_combine[3] = 1
                if 0 not in flag_combine:
                    count_combine_activate += 1
                    break
    return count_combine_activate / count_combine


def two_way_dense_cov(neuron_value, number_of_neuron_layer, boundary):
    neuron_value_activate = neuron_value[:]
    bi_neuron_value_activate = [[0 for _ in range(number_of_neuron_layer)] for __ in range(len(neuron_value))]
    for i in range(len(neuron_value_activate)):
        for j in range(number_of_neuron_layer):
            if neuron_value_activate[i][j] > boundary:
                bi_neuron_value_activate[i][j] = 1
            else:
                bi_neuron_value_activate[i][j] = 0
    count_combine_activate = 0
    count_combine = 0
    for i in range(number_of_neuron_layer):
        for j in range(i + 1, number_of_neuron_layer):
            count_combine += 1
            flag_combine = [0, 0, 0, 0]
            for example in bi_neuron_value_activate:
                if example[i] == 0 and example[j] == 0:
                    flag_combine[0] = 1
                elif example[i] == 0 and example[j] == 1:
                    flag_combine[1] = 1
                elif example[i] == 1 and example[j] == 0:
                    flag_combine[2] = 1
                elif example[i] == 1 and example[j] == 1:
                    flag_combine[3] = 1
            count_combine_activate += sum(flag_combine)
    return count_combine_activate / (count_combine * pow(2, 2))


def two_way_sparse_cov_probability(neuron_value, number_of_neuron_layer, boundary, probability):
    # 神经元结点是否激活
    neuron_value_activate = neuron_value[:]
    bi_neuron_value_activate = [[0 for _ in range(number_of_neuron_layer)] for __ in range(len(neuron_value))]
    for i in range(len(neuron_value_activate)):
        for j in range(number_of_neuron_layer):
            if neuron_value_activate[i][j] > boundary:
                bi_neuron_value_activate[i][j] = 1
            else:
                bi_neuron_value_activate[i][j] = 0

    # 神经元结点值的组合
    count = 0
    count_combine_activate = 0
    count_combine = 0
    count_combine_activate_probability = 0
    for i in range(number_of_neuron_layer):
        for j in range(i + 1, number_of_neuron_layer):
            count_combine += 1
            # print("count_combine = %d" % count_combine)
            flag_combine = [0, 0, 0, 0]
            for example in bi_neuron_value_activate:
                if example[i] == 0 and example[j] == 0:
                    flag_combine[0] = 1
                elif example[i] == 0 and example[j] == 1:
                    flag_combine[1] = 1
                elif example[i] == 1 and example[j] == 0:
                    flag_combine[2] = 1
                elif example[i] == 1 and example[j] == 1:
                    flag_combine[3] = 1
                # print("example[%d] = %f  example[%d] = %f" % (i, example[i], j, example[j]))
            # print(sum(flag_combine))
            if sum(flag_combine) >= (probability * 4):
                count_combine_activate_probability += 1
                # print(count_combine_activate_probability)
    # print("count_combine_activate_probability = %f" % count_combine_activate_probability)
    # print(count_combine_activate_probability)
    return count_combine_activate_probability / count_combine


if __name__ == "__main__":
    # neuron_value_1 = [[1, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1]]
    neuron_value_list = []
    # for k in range(10):
    source_path = r"MNIST_data/testing_data/all_neural_value/layer6_fc_2/class_" + str(8) + "_NeuralValue.txt"
    neuron_number = 84
    temp_neuron_value = load_neural_value(source_path, neuron_number)
    for example_ in temp_neuron_value:
        neuron_value_list.append(example_)
    print(two_way_sparse_cov(neuron_value_list, 84, 0))
    print(two_way_dense_cov(neuron_value_list, 84, 0))
    # print(two_way_dense_cov(neuron_value_1, 4, 0))
    print(two_way_sparse_cov_probability(neuron_value_list, 84, 0, 0.5))
    print(two_way_sparse_cov_probability(neuron_value_list, 84, 0, 0.75))
