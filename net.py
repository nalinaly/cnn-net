# 秦传瑜  2018/11/15 v1
# 本代码主要是基于YJango的https://zhuanlan.zhihu.com/p/22888385学习
# 对冰不语https://blog.csdn.net/xingchenbingbuyu/article/details/53674544的参(chao)考(xi)
# 在win7+pycharm+python3.6上工作正常，不用python3.7是因为这个版本没有自带tensorflow
# 在win7 cmd窗口直接执行 python net.py 也是可以的，建议使用Pycharm调试以加深理解

# python3.6(https://www.python.org/downloads/release/python-367/ 下直接下载Windows x86-64 executable installer)
# 装好后在win7 cmd下直接使用pip install opencv-python，会下载3.4.3.18版本，然而两者并不匹配····
# 可以下载低一点的版本pip install opencv-python==3.2.0.6，然而低版本的opencv并不能自动联想函数，也看不见doc
# 因此多次尝试后，使用pip install opencv-python==3.3.0.10，既可以配合3.6工作也可以联想函数看doc，方便学习
import cv2
# pip install numpy
import numpy as np

class Net:
    def __init__(self):

        # 神经网络的层数及每层的个数，包括输入层，隐藏层，输出层，所以起码有2层
        # 关于输入层与输出层的个数，受限于输入的数据，及想要的结果，通常不会有很大变化。
        # 比如图片识别，一张固定图片大小总共就784像素，第一层输入层的数量就只能是784，如果输出层是需要判断0-9
        # 那么输出也只能是10个，多了也没用。中间的隐藏层的层数设置多少，及每层的个数应该是多少？
        # 随着层数与单层数量的增加，网络的容量上升了，意味着网络能识别出样本里更细节的特征，这既是优势也是不足
        # 优势在于可以分类更加复杂的数据，劣势在于可能造成对训练数据的过拟合，即：本来是样本的一些噪音会被当成
        # 特征学习到，从而在预测真实数据时造成偏差。
        # 那么小网络是否更好呢？也不是，小网络由于容量较小，比较难使用梯度下降等局部方法来训练，虽然小网络的局部
        # 极小值会少一些，也容易收敛到这些极小值，但是这些极小值一般还是很大的，因此实用的时候误差还是较大。
        # 因此：正确的做法是使用较多层数，较多单层个数，然后使用L2正则化等方式防止神经网络过拟合。
        self.layer_neuron_num = np.array([])

        # 激活函数，首先，我们有非线性问题，再多层的卷积运算也是线性的，无法解决非线性问题
        # 其次，反向传播常用梯度下降，需要取导数，而线性函数的导数为常量
        self.activation_function = "sigmoid"

        # 这是为了解决"卡在局部极小值"的问题，所以周期调节学习速率，当然这只是一种解决方法
        self.fine_tune_factor = 1.01

        # 使用梯度下降方式来更新权重，设置一次下降的速度
        self.learning_rate = 0.3

        # 承载神经网络的每一层的权重与偏置
        self.weights = []
        self.bias = []

        # 承载神经网络的每一层神经元
        self.layer = []

        # 目标值，如果是0-9的图片识别的话，就会是一个十元组，下标i为1代表着这个输出值是i
        self.target = np.array([])

        # 计算值与目标值的误差率，用在训练网络时判断是否训练完毕
        self.loss = 0.0

        # 做反向传播的时候，计算值与目标值的差别，反向更新权重值时使用
        self.delta_err = []

        # target - output，用来保存计算完毕后的目标值减去计算值的差值
        self.output_error = np.array([])

    # 初始化神经网络中的神经元，权重，偏置
    def initNet(self, layer_neuron_num_):
        if len(layer_neuron_num_) <= 1:
            print("lack hidden layer!")
            return
        self.layer_neuron_num = layer_neuron_num_
        for i in range(len(self.layer_neuron_num)):
            # 每一层神经元为方便理解与计算，设置为一个多行一列的列向量(但是numpy里没有列向量的概念···)
            self.layer.append(np.ones([self.layer_neuron_num[i], 1]))

        for i in range(len(self.layer_neuron_num) - 1):
            # 主要是线代的矩阵卷积运算来将每一层的值投影到下一层，因为每一层的神经元个数是不一样的
            # 而计算方式为W(i) * X(i)，因此每一层的权重行数为后一层的行数，列数为当前层的行数
            self.weights.append(np.ones([len(self.layer[i + 1]), len(self.layer[i])]))
            self.bias.append(np.ones([len(self.layer[i + 1]), 1]))

    # 随机初始化权重值
    def initWeights(self):
        for weight in self.weights:
            cv2.randn(weight, 0.0, 0.1)

    # 根据传入参数初始化偏置值
    def initBias(self, bias_):
        for i in range(len(self.bias)):
            self.bias[i] += bias_

    # 前向传播函数
    def forward(self):
        for i in range(len(self.layer_neuron_num) - 1):
            # 计算下一层的卷积值(线性计算)
            product = self.weights[i].dot(self.layer[i]) + self.bias[i]
            # 调用激活函数(引入非线性因素)
            self.layer[i + 1] = self.activationFunction(product, self.activation_function)
        # 所有层都计算之后开始计算损失
        self.calcLoss(self.layer[len(self.layer) - 1], self.target)

    # 反向传播计算差值并更新权重与偏置
    def backward(self):
        self.deltaError()
        self.updateWeights()

    # 训练函数，目的是通过反复迭代运算直到权值更新到loss达到阈值为止来获得最终的权值
    def train(self, input_, target_, loss_threshold):
        if len(input_) == 0:
            print("Input is empty!")
            return
        print("Training begin:")

        # 如果输入样本只有一个的情况
        if np.shape(input_)[0] == len(self.layer[0]) and np.shape(input_)[1] == 1:
            self.target = target_
            self.layer[0] = input_
            self.forward()

            num_of_train = 0
            while self.loss > loss_threshold:
                self.backward()
                self.forward()
                num_of_train += 1
                if num_of_train % 100 == 0:
                    print("Training times:", num_of_train, "LOSS:", self.loss)
            print("Training times:", num_of_train, "LOSS:", self.loss, "Success!")

        # 输入样本有多个的情况下，每个样本一次计算加起来的loss和值需要小于loss阈值
        elif np.shape(input_)[0] == len(self.layer[0]) and np.shape(input_)[1] > 1:
            sum_loss = loss_threshold + 1
            batch_train_times = 0
            # 达到设置损失阈值之前反复迭代训练
            while sum_loss > loss_threshold:
                sum_loss = 0
                for i in range(np.shape(input_)[1]):
                    # 这里需要注意的是，获取一个样本或是一个目标值的时候，由于样本与目标值
                    # 是以列向量呈现的，但numpy取完列向量后却以行向量的方式返回，因此需要reshape转成列向量
                    self.layer[0] = input_[:, i].reshape(-1,  1)
                    self.target = target_[:, i].reshape(-1, 1)
                    # 前向传播计算输出值
                    self.forward()
                    # 反向传播更新权重与偏置
                    self.backward()
                    sum_loss += self.loss
                batch_train_times += 1

                # 每10次打印一下loss值，并更新学习速度以避免落入局部极小值
                if batch_train_times % 10 == 0:
                    print("batch_train_times:", batch_train_times, "sum_loss:", sum_loss)
                    self.learning_rate *= self.fine_tune_factor
            print("batch_train_times:", batch_train_times, "sum_loss:", sum_loss, "Success!!!")

    # 测试训练模型的准确率
    def test(self, inputs, targets):
        if len(inputs) == 0 or len(targets) == 0:
            print("inputs or targets is None!")
            return
        if np.shape(inputs)[0] == len(self.layer[0]) and np.shape(targets)[1] == 1:
            predict_num = self.predict_one(inputs)
            # 获取目标列向量里的最大值坐标
            _, target_num = cv2.minMaxLoc(targets)[3]
            print("predict_num:", predict_num, "target_num:", target_num, "LOSS:", self.loss)
        elif np.shape(inputs)[0] == len(self.layer[0]) and np.shape(targets)[1] > 1:
            loss_num = 0
            right_num = 0.0
            for i in range(np.shape(inputs)[1]):
                sample = inputs[:, i].reshape(-1, 1)
                predict_num = self.predict_one(sample)
                loss_num += self.loss

                target = targets[:, i].reshape(-1, 1)
                _, target_num = cv2.minMaxLoc(target)[3]
                print("predict_num:", predict_num, "target_num:", target_num, "LOSS:", self.loss)
                # 预测坐标与预期值一样则认为正确
                if predict_num == target_num:
                    right_num += 1
            # 推理准确个数除以总数得出模型的准确率
            accurary = right_num / np.shape(inputs)[1]
            print("loss_num:", loss_num, "accurary:", accurary)
        else:
            print("Input format doesn't match!")

    # 定义如何计算出计算值与目标值的差异，这里就是差值的2次方，然后求和取平均值
    # 因为输出已经经过sigmoid函数映射到(0,1)空间了,tanh为(-1,1)，所以可以直接当成百分比来看
    def calcLoss(self, output, target):
        if len(target) == 0:
            return
        self.output_error = target - output
        err_sqrare = cv2.pow(self.output_error, 2.0)
        err_sqr_sum = cv2.sumElems(err_sqrare)
        self.loss = err_sqr_sum[0] / float(len(output))

    # sigmoid函数的实现，作用是将任意输入投影到(0,1)的区间里
    @staticmethod
    def sigmoid(x):
        exp_x = cv2.exp(-x)
        fx = 1.0 / (1.0 + exp_x)
        return fx

    # 这也是一个常用的激活函数，输出范围为(-1, 1)
    @staticmethod
    def tanh(x):
        exp_x = cv2.exp(x)
        exp_x_ = cv2.exp(-x)
        fx = (exp_x - exp_x_) / (exp_x + exp_x_)
        return fx

    # 激活函数，主要目的是为整个网络引入非线性特性
    def activationFunction(self, x, func_type):
        fx = np.array([])
        if func_type == "sigmoid":
            fx = self.sigmoid(x)
        elif func_type == "tanh":
            fx = self.tanh(x)
        return fx

    # 激活函数的导数计算公式，用来求权值更新梯度
    def derivativeFunction(self, fx, func_type):
        if func_type == "sigmoid":
            dx = self.sigmoid(fx) * ((1 - self.sigmoid(fx)))
        elif func_type == "tanh":
            tanh_2 = cv2.pow(self.tanh(fx), 2.0)
            dx = 1 - tanh_2
        return dx

    # 计算每一层的差值
    def deltaError(self):
        # pre_alloc list，python里目前还没找到对应c++ resize的合适方法
        self.delta_err = [None] * (len(self.layer) - 1)
        # 因为这里主要是计算差值，是反向传播的，所以需要用reverse
        for i in reversed(range(0, len(self.layer) - 1)):
            # 初始化一个矩阵
            self.delta_err[i] = np.zeros(np.shape(self.layer[i + 1]), type(self.layer[i + 1]))
            # dx是梯度，传入lay[i+1]的值取得f(W[i]X[i])计算的梯度值
            dx = self.derivativeFunction(self.layer[i + 1], self.activation_function)
            # Output layer delta error
            if (i == len(self.layer) - 1 - 1):
                # 最后一层的差值乘以下降梯度
                self.delta_err[i] = dx * (self.output_error)
            # Hidden layer delta error
            else:
                # 中间隐藏层需要先乘以权重后再乘以下降梯度
                delta_weight = np.transpose(self.weights[i + 1]).dot(self.delta_err[i + 1])
                self.delta_err[i] = dx * delta_weight

    # 更新每一层的权重与偏置，具体的计算理论参考http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
    def updateWeights(self):
        for i in range(len(self.weights)):
            delta_layer = np.asarray(self.delta_err[i].dot(np.mat(self.layer[i]).T))
            delta_weights = self.learning_rate * delta_layer
            self.weights[i] = self.weights[i] + delta_weights

            delta_bias = self.learning_rate * self.delta_err[i]
            self.bias[i] = self.bias[i] + delta_bias

    # 使用训练好的模型进行推理
    def predict_one(self, input_):
        if len(input_) == 0:
            print("Input is empty!")
            return
        if np.shape(input_)[0] == len(self.layer[0]) and np.shape(input_)[1] == 1:
            self.layer[0] = input_
            # 进行前向计算
            self.forward()
            layer_out = self.layer[len(self.layer) - 1]
            # 获取输出的列向量里最大值的坐标，如果是数字识别，这个坐标就是推理出的数字
            _, y = cv2.minMaxLoc(layer_out)[3]
            return y
        else:
            print("Input has error!")

    # 批量推理输出推理值
    def predict(self, inputs):
        if np.shape(inputs)[0] == len(self.layer[0]) and np.shape(inputs)[1] > 1:
            predicted_labels = []
            for i in range(np.shape(inputs)[1]):
                sample = inputs[:, i].reshape(-1,  1)
                predicted_label = self.predict_one(sample)
                predicted_labels.append(predicted_label)
            return predicted_labels


# 在此我并没有完全照搬图片识别的数据来进行训练及测试(当然这个框架完全可以用于图片识别)，有几个方面的考虑：
# 一个模型训练需要标签数据(输入样本及预期输出值)，制作图片的标签数据是纯体力活，
# 当然也涉及一点输入的归一化处理，但与训练框架无关，又有较多代码，这里懒得写了。
# 另外一个更重要的因素是：网上的图片识别示例里每个都是784像素，不便于调试理解代码。
# 这个我仅仅是举了一个输入为3个神经元的两个样本，用来方便调试看训练过程中的权重及每一层的输出究竟是什么样的
class train_test:
    # 这个layer_num的首尾取决于input与target，这里是(3,2)，中间隐藏层可以任意添加，
    # 神经元个数也可以随意设置，但是设置太大了内存吃不消
    # layer_num = np.array([3, 88, 8, 2])
    layer_num = np.array([3, 4, 2])
    net = Net()

    # 这六个参数就是模型，所谓的导出模型就是这六个参数(当然高级网络会需要更多的参数，但是简单网络这6个就够了)
    # 训练的目的是在指定layer_num, activation_function, bias, learning_rate, fine_tune_factor
    # 的情况下通过输入input 与 target 及Loss阈值 反复进行计算
    # 得出一个合理的weights(当然weights也可以指定一个初值)
    net.initNet(layer_num)
    # tanh比sigmoid的收敛速度要快一些
    # net.activation_function = "sigmoid"
    net.activation_function = "tanh"
    net.initWeights()
    net.initBias(np.array([0.01]))
    net.learning_rate = 0.2
    net.fine_tune_factor = 1.01

    # 训练样本(标签数据)
    input_ = np.array([[2, 3], [3, 3], [4, 5]])
    target_ = np.array([[0.9, 0.8], [1.0, 1.1]])
    net.train(input_, target_, 0.008)
    # 输出训练好的模型的最终权值与偏置
    print("net.weights", net.weights, "\nnet.bias", net.bias)

    # 测试模型准确率
    # 因为嫌麻烦没有使用数字图片进行识别，因此这个target显的不直观
    # (人脸识别需要更麻烦的输入输出设计，但是基本的训练推理流程是一模一样的)
    # 这里的意思是模型输出为列向量，其中最大值的下标为推理值，仅仅是为了方便调试理解用
    inputs = np.array([[2, 3], [3, 3], [4, 5]])
    targets = np.array([[0.9, 1.2], [1.0, 1.1]])
    net.test(inputs, targets)