import numpy as np


def load_neural_value(fpath, number_of_neuron):
    """
  根据所给文件把神经元输出值从文件中读取
  :param fpath:
  :param number_of_neuron:
  :return:
  """
    with open(fpath, 'r') as f:
        data = f.read()
        numlist = data.split()
        neural_value_ = []
        for number_str in numlist:
            number_float = float(number_str)
            neural_value_.append(number_float)
    neural_value_ = np.array(neural_value_).reshape([-1, number_of_neuron])

    return neural_value_


def get_average_conv_pool(neural_value: list, data_shape: list):
    # 把图片转化为指定格式
    images = np.array(neural_value).reshape(data_shape)
    avg_neural_value = []
    count = 0
    for image in images:  # (这里image是28 * 28 * 6)
        count += 1
        if count % 100 == 0:
            print("当前计算到第%d张" % count)
        avg_temp = []
        for channel in range(data_shape[3]):  # （这里是6，因为一共分成6张图片）
            channel_output = []
            for row_pos in range(data_shape[1]):
                for col_pos in range(data_shape[2]):
                    channel_output.append(image[row_pos][col_pos][channel])
            avg_temp.append(np.mean(channel_output))
        avg_neural_value.append(avg_temp)
    return avg_neural_value
