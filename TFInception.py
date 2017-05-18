import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import csv
from PIL import Image
import inception
from inception import transfer_values_cache
import FCLModel


'''
Workflow of this script:
1. Load data set from train and test csv file into array and float number
2. Caculate the transfer values using pre-trained inception v3 model
3. Build our network model, mainly fully connected layers, and train the model


Several important technical details:
Using random batch in training
Default dropout rate is zero for all layers
Default cost function is vanilla MSE


Parameters need to be initialized before running the code:
PATH_dataset = './Dataset/'
width = 204
height = 204
num_channel = 3
num_train = 2
num_valid = 1
num_test = 2
batch_size = 1
train_epoch = 1
FCLModel_saved = './Saved Model/FCLModelSaved.ckpt'
PATH_data_cache = './Data Cache/'
images_scaled = images_train * 255.0
images_scaled = images_test * 255.0
fc_size0 = 1164             # Number of neurons in fully-connected layer
drop_rate0 = 0            # Drop-out keep rate (0 = no drop out)
fc_size1 = 100
drop_rate1 = 0
fc_size2 = 50
drop_rate2 = 0
fc_size3 = 10
drop_rate3 = 0
cost_func = 0
learn_rate = 1e-4
'''
####################################################################################
# LOAD DATA
##############
PATH_dataset = './Dataset/'
width = 204
height = 204
num_channel = 3
num_train = 2
num_valid = 1
num_test = 2

batch_size = 1
train_epoch = 1

FCLModel_saved = './Saved Model/FCLModelSaved.ckpt'
##############
images_train = np.zeros(shape=[num_train, width, height, num_channel], dtype=float)
images_valid = np.zeros(shape=[num_valid, width, height, num_channel], dtype=float)
images_test = np.zeros(shape=[num_test, width, height, num_channel], dtype=float)
labels_train = np.zeros(shape=[num_train], dtype=float)
labels_valid = np.zeros(shape=[num_valid], dtype=float)
labels_test = np.zeros(shape=[num_test], dtype=float)

# Load training data
with open(PATH_dataset+'train_data.csv') as csv_train:
    count = 0
    reader = csv.reader(csv_train, delimiter=',')
    next(reader, None)  # Skip header
    for row in reader:
        img = Image.open(PATH_dataset + row[0])
        imarray = np.array(img, dtype=np.float_)  # shape: (204,204,3)
        images_train[count, :] = imarray
        labels_train[count] = row[1]

##############
# Load validation data
# ...
##############

# Load test data
with open(PATH_dataset + 'test_data.csv') as csv_test:
    count = 0
    reader = csv.reader(csv_test, delimiter=',')
    next(reader, None)  # Skip header
    for row in reader:
        img = Image.open(PATH_dataset + row[0])
        imarray = np.array(img, dtype=np.float_)  # shape: (204,204,3)
        images_test[count, :] = imarray
        labels_test[count] = row[1]


####################################################################################
# DOWNLOAD THE INCEPTION MODEL
inception.data_dir = './inception/'
inception.maybe_download()
# LOAD INCEPTION MODEL
model = inception.Inception()


####################################################################################
# CALCULATE TRANSFER VALUES
##############
PATH_data_cache = './Data Cache/'
##############
file_path_cache_train = os.path.join(PATH_data_cache, 'inception_train.pkl')
file_path_cache_valid = os.path.join(PATH_data_cache, 'inception_valid.pkl')
file_path_cache_test = os.path.join(PATH_data_cache, 'inception_test.pkl')

print('Processing Inception transfer-values for training-images...')
##############
# # Scale images because Inception needs pixels to be between 0 and 255,
# images_scaled = images_train * 255.0
images_scaled = images_train
##############
# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=images_scaled,
                                              model=model)

print("Processing Inception transfer-values for test-images ...")
##############
# Scale images because Inception needs pixels to be between 0 and 255,
# images_scaled = images_test * 255.0
images_scaled = images_test
##############
# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             images=images_scaled,
                                             model=model)

# print transfer_values_train.shape  # (1,2048)
# print transfer_values_test.shape  # (1,2048)


# HELPER FUNCTION TO PLOT TRANSFER VALUES
def plot_transfer_values(i):
    print("Input image:")

    # Plot the i'th image from the test-set.
    plt.imshow(images_test[i], interpolation='nearest')
    plt.show()

    print("Transfer-values for the image using Inception model:")

    # Transform the transfer-values into an image.
    img = transfer_values_test[i]
    img = img.reshape((32, 64))

    # Plot the image for the transfer-values.
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()

# Visualize the transfer values
# plot_transfer_values(i=0)


####################################################################################
# NEW CLASSIFIER IN TENSORFLOW
# Array length for transfer-values which is stored as variable object for the inception model, i.e. 2048
transfer_len = model.transfer_len
input_dim = transfer_len.value
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[1], name='y_true')


# Neural Network Configuration

# # Wrap the transfer-values as a Pretty Tensor object.
# x_pretty = pt.wrap(x)
# with pt.defaults_scope(activation_fn=tf.nn.relu):
#     y_pred, loss = x_pretty.\
#         fully_connected(size=1024, name='layer_fc1').\
#         softmax_classifier(num_classes=num_classes, labels=y_true)

##############################
# Fully-connected layer 1
fc_size0 = 1164             # Number of neurons in fully-connected layer
drop_rate0 = 0            # Drop-out keep rate (0 = no drop out)

# Fully-connected layer 1
fc_size1 = 100
drop_rate1 = 0

# Fully-connected layer 2
fc_size2 = 50
drop_rate2 = 0

# Fully-connected layer 3
fc_size3 = 10
drop_rate3 = 0
##############################

# Computation Graph
# Fully connected layer 0
layer_fc0 = FCLModel.fc_layer(input=x,
                              num_inputs=input_dim,
                              num_outputs=fc_size0,
                              drop_rate=drop_rate0,
                              use_relu=True)

# Fully connected layer 1
layer_fc1 = FCLModel.fc_layer(input=layer_fc0,
                              num_inputs=fc_size0,
                              num_outputs=fc_size1,
                              drop_rate=drop_rate1,
                              use_relu=True)

# Fully connected layer 2
layer_fc2 = FCLModel.fc_layer(input=layer_fc1,
                              num_inputs=fc_size1,
                              num_outputs=fc_size2,
                              drop_rate=drop_rate2,
                              use_relu=True)

# Fully connected layer 3
layer_fc3 = FCLModel.fc_layer(input=layer_fc2,
                              num_inputs=fc_size2,
                              num_outputs=fc_size3,
                              drop_rate=drop_rate3,
                              use_relu=True)

# Predicted steering angle
y_pred = FCLModel.fc_layer(input=layer_fc3,
                           num_inputs=fc_size3,
                           num_outputs=1,
                           drop_rate=0,
                           use_relu=False)


####################################################################################
# TRAINING NETWORK
cost_func = 0  # default using mean square error as loss function
learn_rate = 1e-4


def random_batch():
    # Number of images (transfer-values) in the training-set.
    num_images = len(transfer_values_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    # We use the transfer-values instead of images as x-values.
    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch


# Function to calculate the Huber loss
def huber_loss(y_true, y_pred, max_grad=1.):
    err = tf.abs(y_true - y_pred, name='abs')
    mg = tf.constant(max_grad, name='max_grad')

    lin = mg * (err - .5 * mg)
    quad = .5 * err * err

    return tf.where(err < mg, quad, lin)


# Cost function
if cost_func == 0:
    # Theory refer to this page: https://davidrosenberg.github.io/ml2015/docs/3a.loss-functions.pdf
    # Mean squared error
    cost = tf.reduce_mean(tf.pow(y_pred - y_true, 2))
elif cost_func == 1:
    # mean squared error (from internet)
    cost = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred),
                                         reduction_indices=[1]))
else:
    # Huber cost function
    cost = huber_loss(y_true, y_pred)

# Optimizer (Adam Optimzer)
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost, global_step)


def compute_error():
    # Put the batch into a dict with the proper names for placeholder variables in the TensorFlow graph
    feed_dict_train = {x: transfer_values_test}

    # Calculate predicted angle
    test_pred = sess.run(y_pred, feed_dict=feed_dict_train)
    test_pred = np.ravel(test_pred)

    # Calculate root mean square error
    error = np.sqrt(np.mean(np.square(labels_test - test_pred)))

    return error


# Train Function
def train_network(num_epoch):
    global FCLModel_saved, batch_size, num_train

    # Number of iterations/mini-batches per epoch
    num_iterations = int(num_train / batch_size)

    # Compute steering error with no training
    error = compute_error()
    print("No Training , Test Error:", error)

    start_time = time.time()

    for j in range(0, num_epoch):
        for i in range(0, num_iterations):
            x_batch, y_true_batch = random_batch()
            # Put the batch into a dict with the proper names for placeholder
            # variables in the TensorFlow graph
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            j_global, _ = sess.run([global_step, optimizer], feed_dict=feed_dict_train)

        # Print status every n epochs
        if j_global % 1 == 0:
            # Calculate the validation error
            test_error = compute_error()
            print("Training Epoch:", j_global, ", Test Error:", test_error)

            # Save model with lower validation error (lower is better)
            if error > test_error:
                # Save the variables to disk.
                save_path = saver.save(sess, FCLModel_saved)
                print("Model saved in file: %s" % save_path)
                error = test_error

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


####################################################################################
# TENSORFLOW SESSION
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    print("Start of neural network training")

    # Initialize graph
    try:
        saver.restore(sess, FCLModel_saved)
        print("Restore from saved session")
    except tf.errors.NotFoundError:
        sess.run(tf.global_variables_initializer())
        print("Initialized global variables")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Train the network
    train_network(train_epoch)

    coord.request_stop()
    coord.join(threads)

print("End of neural network training")