from matplotlib import pyplot as plt

# neuron_value_list_temp = []
#     neuron_number = 28 * 28 * 6
#     for i in range(10):
#         print("加载数据到第%d类" % i)
#         source_path = r"MNIST_data/training_data/all_neural_value/layer1_conv_1/class_" \
#                       + str(i) + "_NeuralValue.txt"
#         temp_neuron_value = load_neural_value(source_path, neuron_number)
#         temp_neuron_value = np.array(temp_neuron_value).reshape([len(temp_neuron_value) * neuron_number])
#         neuron_value_list_temp += list(temp_neuron_value)
#     neuron_value_list_layer1 = get_average_conv_pool(neuron_value_list_temp, [-1, 28, 28, 6])
#
#     neuron_value_list_temp = []
#     neuron_number = 14 * 14 * 6
#     for i in range(10):
#         print("加载数据到第%d类" % i)
#         source_path = r"MNIST_data/training_data/all_neural_value/layer2_maxpool_1/class_" \
#                       + str(i) + "_NeuralValue.txt"
#         temp_neuron_value = load_neural_value(source_path, neuron_number)
#         temp_neuron_value = np.array(temp_neuron_value).reshape([len(temp_neuron_value) * neuron_number])
#         neuron_value_list_temp += list(temp_neuron_value)
#     neuron_value_list_layer2 = get_average_conv_pool(neuron_value_list_temp, [-1, 14, 14, 6])
#
#     neuron_value_list_temp = []
#     neuron_number = 14 * 14 * 16
#     for i in range(10):
#         print("加载数据到第%d类" % i)
#         source_path = r"MNIST_data/training_data/all_neural_value/layer3_conv_2/class_" \
#                       + str(i) + "_NeuralValue.txt"
#         temp_neuron_value = load_neural_value(source_path, neuron_number)
#         temp_neuron_value = np.array(temp_neuron_value).reshape([len(temp_neuron_value) * neuron_number])
#         neuron_value_list_temp += list(temp_neuron_value)
#     neuron_value_list_layer3 = get_average_conv_pool(neuron_value_list_temp, [-1, 14, 14, 16])
#
#     neuron_value_list_temp = []
#     neuron_number = 7 * 7 * 16
#     for i in range(10):
#         print("加载数据到第%d类" % i)
#         source_path = r"MNIST_data/training_data/all_neural_value/layer4_maxpool_2/class_" \
#                       + str(i) + "_NeuralValue.txt"
#         temp_neuron_value = load_neural_value(source_path, neuron_number)
#         temp_neuron_value = np.array(temp_neuron_value).reshape([len(temp_neuron_value) * neuron_number])
#         neuron_value_list_temp += list(temp_neuron_value)
#     neuron_value_list_layer4 = get_average_conv_pool(neuron_value_list_temp, [-1, 7, 7, 16])
#
#     neuron_value_list_layer5 = []
#     neuron_number = 120
#     for i in range(10):
#         source_path = r"MNIST_data/training_data/all_neural_value/layer5_fc_1/class_" + str(i) + "_NeuralValue.txt"
#         temp_neuron_value = load_neural_value(source_path, neuron_number)
#         for example in temp_neuron_value:
#             neuron_value_list_layer5.append(example)
#
#     neuron_value_list_layer6 = []
#     neuron_number = 84
#     for i in range(10):
#         source_path = r"MNIST_data/training_data/all_neural_value/layer6_fc_2/class_" + str(i) + "_NeuralValue.txt"
#         temp_neuron_value = load_neural_value(source_path, neuron_number)
#         for example in temp_neuron_value:
#             neuron_value_list_layer6.append(example)
#
#     for i in range(len(neuron_value_list_layer6)):
#         global_neuron_value_list.append(list(neuron_value_list_layer1[i]) + list(neuron_value_list_layer2[i])
#                                         + list(neuron_value_list_layer3[i]) + list(neuron_value_list_layer4[i])
#                                         + list(neuron_value_list_layer5[i]) + list(neuron_value_list_layer6[i]))


if __name__ == "__main__":
    cov_cumulative = [1, 2, 3, 4, 5]
    x = [1, 2, 3, 4, 5]
    y = [4, 5, 6, 7, 8]

    fig = plt.figure(dpi=80)
    plt.title("Min Test Coverage Rate")
    plt.xlabel("number of examples")
    plt.ylabel("neuron coverage rate")
    plt.plot(x, cov_cumulative, marker='o')

    # ax = plt.gca()
    # ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    # ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    plt.plot(x, y)
    # 第二个参数是写在哪里
    for xy in zip(x, cov_cumulative):
        plt.annotate("(%s,%s)" % xy, xy=xy, xytext=(-20, 10), textcoords='offset points', )
    plt.legend(["min test coverage", "coverage increase rate"], loc="lower right")
    plt.show()