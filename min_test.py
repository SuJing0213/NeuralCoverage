from deepgauge_cov import get_boundary, k_multisection_neuron_coverage
from deepxplore_cov import neuron_coverage, calculate_cumulative_ncov_rate
from input_data import load_neural_value, get_average_conv_pool
import numpy as np
from matplotlib import pyplot as plt


def min_test_example_cov(neuron_value: list, activate_bound: float, expected_cov: float, number_of_neuron: int):
    """
    DeepXplore中提出的神经元覆盖率
    :param neuron_value: 神经元值 -1 * 样本数
    :param activate_bound: 激活阈值
    :param expected_cov: 对于该层神经元覆盖率的最大值
    :param number_of_neuron: 神经元个数
    :return: 需要多少个样本，所选样本的数量, 覆盖率累计增长
    """
    # 已激活的神经元
    activated_neuron = [0 for _ in range(number_of_neuron)]
    # 利用贪心算法所选择的样本
    example_selected = []
    example_number = 0
    # 当前覆盖率
    temp_cov = 0
    cov_cum = [0]

    while temp_cov < expected_cov:
        max_count = 0
        example_index = -1
        # 对于每一个样本
        for i in range(len(neuron_value)):
            count = 0
            # 对于每一个神经元节点
            for k in range(number_of_neuron):
                # 当前神经元节点被激活且未曾被别的样本激活（激活的增量）
                if neuron_value[i][k] > activate_bound and activated_neuron[k] == 0:
                    # 记录激活增量
                    count += 1
            # 所有样本中挑激活增量最大的样本记录下来
            if count > max_count:
                max_count = count
                example_index = i
        example_selected.append(example_index)
        example_number += 1
        # 使用选择的覆盖率增长最大的样本（索引为example_index）更新activated_neuron
        for k in range(number_of_neuron):
            if neuron_value[example_index][k] > activate_bound:
                activated_neuron[k] = 1
        temp_cov = sum(activated_neuron)/number_of_neuron
        cov_cum.append(temp_cov)
        print("当前选取了 %d 张图片，覆盖率达到了 %f" % (example_number, temp_cov))
    return example_number, example_selected, cov_cum


def min_test_example_nbcov(neuron_value: list, bound: list, expected_cov: float, number_of_neuron: int):
    """
    NBCov中提出的神经元覆盖率
    :param neuron_value: 神经元值 -1 * 样本数
    :param bound: 每个神经元结点的阈值
    :param expected_cov: 对于该层神经元覆盖率的最大值
    :param number_of_neuron: 神经元个数
    :return: 需要多少个样本，所选样本的数量
    """
    # 已激活的神经元
    activated_neuron = [[0 for _ in range(2)] for _ in range(number_of_neuron)]
    # 利用贪心算法所选择的样本
    example_selected = []
    example_number = 0
    # 当前覆盖率
    temp_cov = 0

    while temp_cov < expected_cov:
        max_count = 0
        example_index = -1
        # 对于每一个样本
        for i in range(len(neuron_value)):
            count = 0
            # 对于每一个神经元节点
            for k in range(number_of_neuron):
                # 当前神经元节点被激活且未曾被别的样本激活（激活的增量）
                if neuron_value[i][k] > bound[k]["max"] and activated_neuron[k][1] == 0:
                    count += 1
                if neuron_value[i][k] < bound[k]["min"] and activated_neuron[k][0] == 0:
                    count += 1
            # 所有样本中挑激活增量最大的样本记录下来
            if count > max_count:
                max_count = count
                example_index = i
        example_selected.append(example_index)
        example_number += 1
        # 使用选择的覆盖率增长最大的样本（索引为example_index）更新activated_neuron
        for k in range(number_of_neuron):
            if neuron_value[example_index][k] > bound[k]["max"]:
                activated_neuron[k][1] = 1
            if neuron_value[example_index][k] < bound[k]["min"]:
                activated_neuron[k][0] = 1
        # print(np.array(activated_neuron).shape)
        temp_cov = np.sum(np.array(activated_neuron)) / (number_of_neuron * 2)

        print("当前选取了 %d 张图片，覆盖率达到了 %f" % (example_number, temp_cov))
    return example_number, example_selected


def min_test_example_snacov(neuron_value: list, bound: list, expected_cov: float, number_of_neuron: int):
    """
    SNACov神经元覆盖率
    :param neuron_value: 神经元值 -1 * 样本数
    :param bound: 每个神经元结点的阈值
    :param expected_cov: 对于该层神经元覆盖率的最大值
    :param number_of_neuron: 神经元个数
    :return: 需要多少个样本，所选样本的数量
    """
    # 已激活的神经元
    activated_neuron = [0 for _ in range(number_of_neuron)]
    # 利用贪心算法所选择的样本
    example_selected = []
    example_number = 0
    # 当前覆盖率
    temp_cov = 0

    while temp_cov < expected_cov:
        max_count = 0
        example_index = -1
        # 对于每一个样本
        for i in range(len(neuron_value)):
            count = 0
            # 对于每一个神经元节点
            for k in range(number_of_neuron):
                # 当前神经元节点被激活且未曾被别的样本激活（激活的增量）
                if neuron_value[i][k] > bound[k]["max"] and activated_neuron[k] == 0:
                    count += 1
            # 所有样本中挑激活增量最大的样本记录下来
            if count > max_count:
                max_count = count
                example_index = i
        example_selected.append(example_index)
        example_number += 1
        # 使用选择的覆盖率增长最大的样本（索引为example_index）更新activated_neuron
        for k in range(number_of_neuron):
            if neuron_value[example_index][k] > bound[k]["max"]:
                activated_neuron[k] = 1
        # print(np.array(activated_neuron).shape)
        temp_cov = np.sum(activated_neuron) / number_of_neuron

        print("当前选取了 %d 张图片，覆盖率达到了 %f" % (example_number, temp_cov))
    return example_number, example_selected


def min_test_example_top_k_cov(neuron_value: list, expected_cov: float, number_of_neuron: int, k_value: int):
    """
    top_k_cov神经元覆盖率
    :param neuron_value: 神经元值 -1 * 样本数
    :param expected_cov: 对于该层神经元覆盖率的最大值
    :param number_of_neuron: 神经元个数
    :param k_value: top_k
    :return: 需要多少个样本，所选样本的数量
    """
    # 已激活的神经元
    activated_neuron = [0 for _ in range(number_of_neuron)]
    # 利用贪心算法所选择的样本
    example_selected = []
    example_number = 0
    # 当前覆盖率
    temp_cov = 0

    while temp_cov < expected_cov:
        max_count = 0
        example_index = -1          # 对于每一个样本
        for i in range(len(neuron_value)):
            count = 0
            neuron_value_to_list = neuron_value[i].tolist()  # 将矩阵转为列表
            for k in range(k_value):                                         # 选出最大的k个神经元结点,得出top-k的下标
                max_index = neuron_value_to_list.index(max(neuron_value_to_list))    # 得到最大值的索引
                neuron_value_to_list[max_index] = -1
                if activated_neuron[max_index] == 0:                          # 如果该索引没被标记过则标记
                    count += 1
            if count > max_count:                                             # 所有样本中挑激活增量最大的样本记录下来
                max_count = count
                example_index = i
            if max_count == k_value:
                break
        example_selected.append(example_index)
        example_number += 1

        # 使用选择的覆盖率增长最大的样本（索引为example_index）更新activated_neuron
        neuron_value_to_list = neuron_value[example_index].tolist()  # 将矩阵转为列表
        for k in range(k_value):
            max_index = neuron_value_to_list.index(max(neuron_value_to_list))  # 得到最大值的索引
            neuron_value_to_list[max_index] = -1
            activated_neuron[max_index] = 1
        temp_cov = sum(activated_neuron) / number_of_neuron
        print("当前选取了 %d 张图片，覆盖率达到了 %f" % (example_number, temp_cov))

    return example_number, example_selected


def min_test_example_k_multisection_cov(neuron_value: list,
                                        boundary: list,
                                        expected_cov: float,
                                        number_of_neuron: int,
                                        number_of_section: int):
    """
    DeepXplore中提出的神经元覆盖率
    :param boundary: 所有神经元的最大最小值
    :param neuron_value: 神经元值 -1 * 样本数
    :param expected_cov: 对于该层神经元覆盖率的最大值
    :param number_of_neuron: 神经元个数
    :param number_of_section: 分成k份
    :return: 需要多少个样本，所选样本的数量
    """

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

    # 已激活的神经元
    activated_neuron = [[0 for _ in range(number_of_section)] for _ in range(number_of_neuron)]
    # 利用贪心算法所选择的样本
    example_selected = []
    example_number = 0
    # 当前覆盖率
    temp_cov = 0
    while temp_cov < expected_cov:
        max_count = 0
        example_index = -1
        # 对于每一个样本
        for i in range(len(neuron_value)):
            count = 0
            # 对于每一个神经元节点
            for k in range(number_of_neuron):
                for m in range(number_of_section):
                    # 当前神经元节点被激活且未曾被别的样本激活（激活的增量）
                    if k_section_bound[k][m] <= neuron_value[i][k] < k_section_bound[k][m+1] \
                            and activated_neuron[k][m] == 0:
                        count += 1
                        break
            # 所有样本中挑激活增量最大的样本记录下来
            if count > max_count:
                max_count = count
                example_index = i
        example_selected.append(example_index)
        example_number += 1
        # 使用选择的覆盖率增长最大的样本（索引为example_index）更新activated_neuron
        for k in range(number_of_neuron):
            for m in range(number_of_section):
                # 当前神经元节点被激活且未曾被别的样本激活（激活的增量）
                if k_section_bound[k][m] <= neuron_value[example_index][k] < k_section_bound[k][m+1]:
                    activated_neuron[k][m] = 1
        # print(np.array(activated_neuron).shape)
        temp_cov = np.sum(np.array(activated_neuron)) / (number_of_neuron * number_of_section)
        print("当前选取了 %d 张图片，覆盖率达到了 %f" % (example_number, temp_cov))
    return example_number, example_selected


if __name__ == "__main__":

    # -------------------------------NCov在所有神经网络上用多少样本可以达到最大覆盖率-------------------------------------
    global_neuron_number = 6+6+16+16+120+84
    act_bound = 0
    source_path = r"MNIST_data/training_data/all_neural_value/layer1_conv_1/All_class_avg_NeuralValue.txt"
    global_neuron_value_list = load_neural_value(source_path, global_neuron_number)
    # print(np.array(global_neuron_value_list).shape)
    # result_file = r"MNIST_data/training_data/all_neural_value/layer1_conv_1/All_class_avg_NeuralValue.txt"
    # with open(result_file, "w") as f:
    #     global_neuron_value_list = np.array(global_neuron_value_list).reshape(
    #         [len(global_neuron_value_list)*global_neuron_number])
    #     for value in global_neuron_value_list:
    #         print(value, end='    ', file=f)
    #     print("\n", file=f)
    cov = neuron_coverage(global_neuron_value_list, global_neuron_number, act_bound)
    print(cov)
    cumulative_cov_rate = calculate_cumulative_ncov_rate(global_neuron_value_list, global_neuron_number, act_bound)
    output_path = r"MNIST_data/training_data/all_neural_value/layer1_conv_1/cumulative_cov_rate.txt"
    with open(output_path, "w") as f:
        for elem in cumulative_cov_rate:
            print(elem, file=f)

    fig = plt.figure(dpi=80)  # figsize=(20, 8),
    plt.title("Cumulative Coverage Rate")
    plt.xlabel("number of examples")
    plt.ylabel("neuron coverage rate")
    x = [i for i in range(55001)]
    plt.plot(x, cumulative_cov_rate)
    y = []
    for i in range(55001):
        y.append(i / 55000)
    plt.plot(x, y)
    plt.legend(["coverage", "coverage increase rate"], loc="lower right")
    plt.show()

    # num_example, index_example, cov_cumulative = min_test_example_cov(global_neuron_value_list,
    #                                                                   act_bound, cov, global_neuron_number)
    # fig = plt.figure(dpi=80)
    # plt.title("Min Test Coverage Rate")
    # plt.xlabel("number of examples")
    # plt.ylabel("neuron coverage rate")
    # x = [i for i in range(num_example+1)]
    # plt.plot(x, cov_cumulative, marker='o')
    # for i in range(len(cov_cumulative)):
    #     cov_cumulative[i] = round(cov_cumulative[i], 3)
    # y = []
    # for i in range(num_example+1):
    #     y.append(i / num_example)
    # plt.plot(x, y)
    # # 第二个参数是写在哪里
    # for xy in zip(x, cov_cumulative):
    #     plt.annotate("(%s,%s)" % xy, xy=xy, xytext=(-20, 10), textcoords='offset points')
    # plt.legend(["min test coverage", "coverage increase rate"], loc="lower right")
    # plt.show()

    # neuron_value_list = []
    # neuron_number = 120
    # for j in range(10):
    #     source_path = r"MNIST_data/testing_data/all_neural_value/layer5_fc_1/class_" + str(j) + "_NeuralValue.txt"
    #     temp_neuron_value = load_neural_value(source_path, neuron_number)
    #     for example_ in temp_neuron_value:
    #         neuron_value_list.append(example_)
    #
    # neuron_value_train = []
    # for j in range(10):
    #     source_path = r"MNIST_data/training_data/all_neural_value/layer5_fc_1/class_" + str(j) + "_NeuralValue.txt"
    #     temp_neuron_value = load_neural_value(source_path, neuron_number)
    #     for example_ in temp_neuron_value:
    #         neuron_value_train.append(example_)

    # act_bound = 0
    # cov = neuron_coverage(neuron_value_list, 28*28*6, act_bound)
    # print(act_bound)
    # print(min_test_example_cov(neuron_value_list, act_bound, cov, 28*28*6))

    # print(min_test_example_nbcov(neuron_value_list, get_boundary(neuron_value_train, neuron_number),
    #                              0.0797, neuron_number))

    # print(min_test_example_snacov(neuron_value_list, get_boundary(neuron_value_train, neuron_number),
    #                               0.155, neuron_number))

    # print(min_test_example_top_k_cov(neuron_value_list, 0.8416666666666667, neuron_number, 1))
    # print(min_test_example_top_k_cov(neuron_value_list, 0.925, neuron_number, 2))
    # print(min_test_example_top_k_cov(neuron_value_list, 0.9333333333333333, neuron_number, 3))

    # print(min_test_example_k_multisection_cov(neuron_value_list, get_boundary(neuron_value_train, 120), 0.8866, 120, 100))
    # print(min_test_example_k_multisection_cov(neuron_value_list, get_boundary(neuron_value_train, 84), 1, 84, 1000))