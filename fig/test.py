"""
测试network
import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
"""

'''
#加入交叉熵
import mnist_loader
import network2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10],cost = network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,monitor_evaluation_accuracy=True)
'''

import mnist_loader
import network2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10],cost = network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.1, lmbda=5.0, evaluation_data=test_data,monitor_evaluation_accuracy=True)