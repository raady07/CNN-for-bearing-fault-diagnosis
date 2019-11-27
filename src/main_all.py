
# taking all rpm samples together - each rpm files has all 4 classes - normal, BCI, BCO,BCR
# all samples are mixed and 25 : 75 are taken as
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

# choose which frequency to test
freq = 48000                        #12000,  48000
# choose which crack size to test
cracksize = 14                      # 7, 14, 21
# choose which rpm to be used for training , the remaining 3 will be used for testing
# 1 - rpm_1797, 2 - rpm_1772, 3 - rpm_1748, 4 - rpm_1724
whichonein4 = 1                     # 1, 2, 3, 4

class dimensions:
    # folder to input data
    if freq == 12000:
        folder = 'D:\\Lab Project Files\\MY1DCNN\\Data\\raw\\12KHz\\'
    elif freq == 48000:
        folder = 'D:\\Lab Project Files\\MY1DCNN\\Data\\raw\\48KHz\\'

    num_len = 4096
    # loading the train and test files based on the choice
    if cracksize == 7:
        if whichonein4 == 1:
            files = ['rpm_1797_7mm.txt', 'rpm_1772_7mm.txt', 'rpm_1748_7mm.txt', 'rpm_1724_7mm.txt']
        elif whichonein4 == 2:
            files = ['rpm_1772_7mm.txt', 'rpm_1748_7mm.txt', 'rpm_1724_7mm.txt', 'rpm_1797_7mm.txt']
        elif whichonein4 == 3:
            files = ['rpm_1748_7mm.txt', 'rpm_1724_7mm.txt', 'rpm_1797_7mm.txt', 'rpm_1772_7mm.txt']
        elif whichonein4 == 4:
            files = ['rpm_1724_7mm.txt', 'rpm_1797_7mm.txt', 'rpm_1772_7mm.txt', 'rpm_1748_7mm.txt']

    elif cracksize == 14:
        if whichonein4 == 1:
            files = ['rpm_1797_14mm.txt', 'rpm_1772_14mm.txt', 'rpm_1748_14mm.txt', 'rpm_1724_14mm.txt']
        elif whichonein4 == 2:
            files = ['rpm_1772_14mm.txt', 'rpm_1748_14mm.txt', 'rpm_1724_14mm.txt', 'rpm_1797_14mm.txt']
        elif whichonein4 == 3:
            files = ['rpm_1748_14mm.txt', 'rpm_1724_14mm.txt', 'rpm_1797_14mm.txt', 'rpm_1772_14mm.txt']
        elif whichonein4 == 4:
            files = ['rpm_1724_14mm.txt', 'rpm_1797_14mm.txt', 'rpm_1772_14mm.txt', 'rpm_1748_14mm.txt']

    elif cracksize == 21:
        if whichonein4 == 1:
            files = ['rpm_1797_21mm.txt', 'rpm_1772_21mm.txt', 'rpm_1748_21mm.txt', 'rpm_1724_21mm.txt']
        elif whichonein4 == 2:
            files = ['rpm_1772_21mm.txt', 'rpm_1748_21mm.txt', 'rpm_1724_21mm.txt', 'rpm_1797_21mm.txt']
        elif whichonein4 == 3:
            files = ['rpm_1748_21mm.txt', 'rpm_1724_21mm.txt', 'rpm_1797_21mm.txt', 'rpm_1772_21mm.txt']
        elif whichonein4 == 4:
            files = ['rpm_1724_21mm.txt', 'rpm_1797_21mm.txt', 'rpm_1772_21mm.txt', 'rpm_1748_21mm.txt']

    # selecting the parameters of the deep CNN
    # input dimension of the CNN
    input_width, input_height, input_depth   = 1, num_len, 1
    # kernel sizes of first layer
    conv1_filterwidth, conv1_filterheight, conv1_filters = 1, 5, 4 # 16 filters of 5x5
    # kernel sizes of the second layer
    conv2_filterwidth, conv2_filterheight, conv2_filters = 1, 5, 4 # 16 filters of 5x5
    # pooling layer dimensions
    pool_filterwidth, pool_filterheight = 1, 2
    # strides in conv layer
    conv_stridewidth, conv_strideheight = 1, 1  # strides in convolution layer
    # pool strides in pooling layer
    pool_stridewidth, pool_strideheight = 1, 2  # strides in pooling layer / spatial extent
    # number of nodes in fully connected node
    # no particular choice - optimally choosen
    fc_nodes   = 64
    # number of output classes
    No_Classes = 4
    # batch size for each iterations
    BATCH_SIZE = 128
    # dropuout 50% 
    dropoutrate_train = 0.5
    # for testing no dropout
    dropoutrate_test = 1.0
    # choosen experimentally where the alogorthm is converging
    no_iterations = 40
    # k fold cross validation
    cross_validation = 10
    #operations
    #paddings = 'SAME' # obtain the same size as input
    paddings = 'VALID'

    # defining the dimensions of the CONVNET layer
    def conv_output(W1, H1, F_width, F_height, paddings, S_W, S_H):
        if paddings == 'SAME':
            P_width  = (F_width - 1)/2
            P_height = (F_height- 1)/2
        elif paddings == 'VALID':
            P_width = 0.00
            P_height= 0.00
        W2 = int((W1 - F_width  + 2*P_width)/S_W + 1)
        H2 = int((H1 - F_height + 2*P_height)/S_H + 1)
        return W2, H2
    # defining the dimensions of the pooling layer
    def pool_output(W1, H1, F_width, F_height, S_W, S_H):
        W2 = int((W1 - F_width) / S_W + 1)
        H2 = int((H1 - F_height) / S_H + 1)
        return W2, H2
    # all weights in the network are intialized randomly
    def init_weigths(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))
    # layer 1 initial weight
    layer1_weights = init_weigths([conv1_filterwidth,
                                   conv1_filterheight,
                                   input_depth,
                                   conv1_filters])
    # layer 1 initial biases
    layer1_biases  = tf.Variable(tf.zeros([conv1_filters]))
    # layer 2 initial weights
    layer2_weights = init_weigths([conv2_filterwidth,
                                   conv2_filterheight,
                                   conv1_filters,
                                   conv2_filters])
    # layer 2 initial biases
    layer2_biases  = tf.Variable(tf.zeros([conv2_filters]))

    # sizes of each CONVNET layer after applying conv and pool operations
    layer1_width, layer1_height = conv_output(input_width, input_height, conv1_filterwidth, conv1_filterheight, paddings, conv_stridewidth, conv_strideheight)
    layer1_width, layer1_height = pool_output(layer1_width, layer1_height, pool_filterwidth, pool_filterheight, pool_stridewidth, pool_strideheight)
    layer2_width, layer2_height = conv_output(layer1_width, layer1_height, conv2_filterwidth, conv2_filterheight, paddings, conv_stridewidth, conv_strideheight)
    layer2_width, layer2_height = pool_output(layer2_width, layer2_height, pool_filterwidth, pool_filterheight, pool_stridewidth, pool_strideheight)
    # intializing the nodes of the fully connected layer based on the dimensions of the last CONVNet layer
    fc_weights = tf.Variable(tf.truncated_normal([conv2_filters*layer2_width*layer2_height, fc_nodes], stddev=0.01))
    fc_biases  = tf.Variable(tf.constant(1.0, shape=[fc_nodes]))
    # intializing the nodes for softmax layers 
    Softmax_weights = tf.Variable(tf.truncated_normal([fc_nodes, No_Classes], stddev=0.01))
    Softmax_biases  = tf.Variable(tf.constant(1.0, shape=[No_Classes]))

# load data 
# data is preprocessed and saved in txt files
def load_data(model):
    folder = model.folder
    files =  model.files
    samples = model.input_height
    x_datatrain = []
    x_datatest  = []
    y_datatrain = []
    y_datatest  = []
    filename = folder + files[0]
    data = np.loadtxt(filename, unpack= True)
    x_datatrain.extend(data[:,0:samples])
    y_datatrain.extend(data[:,samples:])
    for i in range(1,4):
        filename = folder + files[i]
        data = np.loadtxt(filename, unpack = True)
        x_datatest.extend(data[:,0:samples])
        y_datatest.extend(data[:,samples:])
    return x_datatrain, x_datatest, y_datatrain, y_datatest # x - data , y - labels

# reshape data in to dimensions that satisfy the CNN input nodes
def reshapedata(data_train,data_test,model):
    data_train = np.reshape(data_train,[-1,model.input_width, model.input_height, model.input_depth])
    data_test  = np.reshape(data_test,[-1, model.input_width, model.input_height, model.input_depth])
    return data_train,data_test

# select the data and group in to batch for batch processing single parameter update
def batchdata(data,label, batchsize):
    # generate random number required to batch data
    order_num = random.sample(range(1, len(data)), batchsize)
    data_batch = []
    label_batch = []
    for i in range(len(order_num)):
        data_batch.append(data[order_num[i-1]])
        label_batch.append(label[order_num[i-1]])
    return data_batch, label_batch

# proposed CNN architecture model
def mycnn(x,model,p_keep_conv):
    CS_w, CS_h = model.conv_stridewidth, model.conv_strideheight # stride in convolution
    PF_w, PF_h = model.pool_filterwidth, model.pool_filterheight # pooling filter size
    PS_w, PS_h = model.pool_stridewidth, model.pool_strideheight

    conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(x, model.layer1_weights, strides=[1, CS_w, CS_h, 1], padding= model.paddings), model.layer1_biases))
    layer1 = tf.nn.max_pool(conv1, ksize=[1, PF_w, PF_h, 1], strides=[1, PS_w, PS_h, 1], padding='SAME')

    conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(layer1, model.layer2_weights, strides=[1, CS_w, CS_h, 1], padding=model.paddings), model.layer2_biases))
    layer2 = tf.nn.max_pool(conv2, ksize=[1, PF_w, PF_h, 1], strides=[1, PS_w, PS_h, 1], padding='SAME')

    shape = layer2.get_shape().as_list()
    reshape = tf.reshape(layer2, [shape[0], shape[1] * shape[2] * shape[3]])
    FClayer = tf.nn.relu(tf.add(tf.matmul(reshape, model.fc_weights), model.fc_biases))
    FClayer = tf.nn.dropout(FClayer,p_keep_conv)
    softmax_layer = tf.add(tf.matmul(FClayer, model.Softmax_weights), model.Softmax_biases)
    return softmax_layer
###############main#####################
sess = tf.Session()
# the dimension of the network
model = dimensions()
# loading data 
data_train, data_test, label_train, label_test =  load_data(model)
# splitting the data in to train and test (25:75 as samples from one rpm for train and 3 rpm data for test
data_train, data_test, = reshapedata(data_train, data_test, model)
# input output placeholders
x  = tf.placeholder(tf.float32, [model.BATCH_SIZE, model.input_width,model.input_height,model.input_depth]) # last column = 1 -> channels here is 1 , for RGB = 3
y_ = tf.placeholder(tf.float32, [model.BATCH_SIZE, model.No_Classes])
# dropout rate
p_keep_conv = tf.placeholder("float")
# proposed CNN
y  = mycnn(x,model, p_keep_conv)
# loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# train step
train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)
# prediction
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# confusion matrix
confusion = tf.confusion_matrix(labels=tf.argmax(y_,1),predictions=tf.argmax(y,1),num_classes=model.No_Classes,dtype=tf.int32)
# compute accuracy formula
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
lossfun = np.zeros(model.no_iterations)
sess.run(tf.global_variables_initializer())

# running the training 
for i in range(model.no_iterations):
    image_batch, label_batch = batchdata(data_train, label_train, model.BATCH_SIZE)
    epoch_loss = 0
    for j in range(model.BATCH_SIZE):
        sess.run(train_step, feed_dict={x: image_batch, y_: label_batch, p_keep_conv:model.dropoutrate_train})
        c = sess.run( cost, feed_dict={x: image_batch, y_: label_batch, p_keep_conv: model.dropoutrate_train})
        epoch_loss += c
    lossfun[i] = epoch_loss
    print('Epoch',i,'completed out of',model.no_iterations,'loss:',epoch_loss )

# testing the training accuracy
image_batch, label_batch = batchdata(data_test,label_test,model.BATCH_SIZE)
print('train accuracy: ')
print((sess.run(accuracy, feed_dict={x: image_batch, y_: label_batch, p_keep_conv:model.dropoutrate_test}))*100)
# print the loss curve
# we can observe if the network is properly trained if the graph satisfy the P5 plot

# running the testing
b = np.zeros([1,4])
cc = 0
# accuracy is computed on k fold cross validataion
for i in range(model.cross_validation):
    image_batch_test, label_batch_test = batchdata(data_test,label_test,model.BATCH_SIZE)
    c = sess.run(confusion, feed_dict={x: image_batch_test, y_: label_batch_test, p_keep_conv:model.dropoutrate_test})
    cc += c
# final test accuracy of the individual classes
print(cc)
b[0,0] = (cc[0,0]/np.sum(cc[0]))*100
b[0,1]  = (cc[1,1]/np.sum(cc[1]))*100
b[0,2]  = (cc[2,2]/np.sum(cc[2]))*100
b[0,3]  = (cc[3,3]/np.sum(cc[3]))*100

# overall accuracy of the proposed algorithm
print('Test Accuracy:')
print(b)
print(np.mean(b))
