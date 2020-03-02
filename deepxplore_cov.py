from input_data import load_neural_value


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


if __name__ == '__main__':

    neuron_value_list = []
    # for i in range(10):
    source_path = r"MNIST_data/testing_data/all_neural_value/layer6_fc_2/class_" + str(9) + "_NeuralValue.txt"
    neuron_number = 84
    temp_neuron_value = load_neural_value(source_path, neuron_number)
    for example_ in temp_neuron_value:
        neuron_value_list.append(example_)

        # neuron_value_test = [[0, 0.75, 0, 0], [0, 0.26, 0, 0.49], [0.2, 0, 0.76, 0], [0, 0, 0, 0]]

    # cov = neuron_coverage(neuron_value_list, 10*10*16, act_bound)
    # print("active_bound = %f cov = %f" % (act_bound, cov))
    act_bound = 0
    cov = neuron_coverage(neuron_value_list, 84, act_bound)
    print("active_bound = %f cov = %f" % (act_bound, cov))
