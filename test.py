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