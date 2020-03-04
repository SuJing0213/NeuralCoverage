import numpy as np
import tensorflow as tf
import sys


np.set_printoptions(precision=8, suppress=True, threshold=sys.maxsize)


def get_neural_value(source_path, result_path, tensor_name, number_of_value):
    """
    给出图片数据集，输出每张图片经过网络后，每个位置上的神经元输出值
    """
    with open(source_path, 'r') as source_file:
        data = source_file.read()
        image = []
        numlist = data.split()
        for number_str in numlist:
            number_float = float(number_str)
            image.append(number_float)
    images = np.array(image).reshape([-1, 784])
    label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    sess = tf.Session()
    saver = tf.train.import_meta_graph(r'Model/model.ckpt.meta')
    saver.restore(sess, r"Model/model.ckpt")
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    with open(result_path, 'w') as result_file:
        for k in range(len(images)):
            picture = np.array(images[k]).reshape([1, 28, 28, 1])
            feed_dict = {x: picture, y_: label, keep_prob: 1.0}
            tensor = graph.get_tensor_by_name(tensor_name)
            layer_output = sess.run(tensor, feed_dict)
            # print(np.array(layer_output).shape)
            # input()
            layer_output = np.array(layer_output).reshape([number_of_value])
            for value in layer_output:
                print(value, end='    ', file=result_file)
            print("\n", file=result_file)


if __name__ == "__main__":

    for i in range(10):
        Source_path = r'MNIST_data/training_data/train_images_class_' + str(i) + '.txt'

        # 第一层卷积的输出
        Result_path = r'MNIST_data/training_data/all_neural_value/layer1_conv_1/class_' + \
                      str(i) + '_NeuralValue.txt'
        get_neural_value(source_path=Source_path, result_path=Result_path,
                         tensor_name="layer1-conv1/conv1:0", number_of_value=28*28*6)


        # 第二层池化的输出
        # Result_path = r'MNIST_data/training_data/all_neural_value/layer2_maxpool_1/class_' + \
        #               str(i) + '_NeuralValue.txt'
        # get_neural_value(source_path=Source_path, result_path=Result_path,
        #                  tensor_name="layer2-pool1/pool1:0", number_of_value=14*14*6)

        # 第三层卷积的输出
        # Result_path = r'MNIST_data/training_data/all_neural_value/layer3_conv_2/class_' + \
        #               str(i) + '_NeuralValue.txt'
        # get_neural_value(source_path=Source_path, result_path=Result_path,
        #                  tensor_name="layer3-conv2/conv2:0", number_of_value=14*14*16)

        # 第四层池化的输出
        # Result_path = r'MNIST_data/training_data/all_neural_value/layer4_maxpool_2/class_' + \
        #               str(i) + '_NeuralValue.txt'
        # get_neural_value(source_path=Source_path, result_path=Result_path,
        #                  tensor_name="layer4-pool2/pool2:0", number_of_value=7*7*16)

        # 第五层全连接的输出
        # Result_path = r'MNIST_data/training_data/all_neural_value/layer5_fc_1/class_' + \
        #               str(i) + '_NeuralValue.txt'
        # get_neural_value(source_path=Source_path, result_path=Result_path,
        #                  tensor_name="layer5-fc1/fc1:0", number_of_value=120)

        # 第六层全连接的输出
        # Result_path = r'MNIST_data/training_data/all_neural_value/layer6_fc_2/class_' + \
        #               str(i) + '_NeuralValue.txt'
        # get_neural_value(source_path=Source_path, result_path=Result_path,
        #                  tensor_name="layer6-fc2/fc2:0", number_of_value=84)

        # # 第七层全连接的输出
        # Result_path = r'MNIST_data/training_data/all_neural_value/layer7_fc_3/class_' + \
        #               str(i) + '_NeuralValue.txt'
        # get_neural_value(source_path=Source_path, result_path=Result_path,
        #                  tensor_name="layer7-fc3/output:0", number_of_value=10)
