##from create_sentiment_featuresets import create_feature_sets_and_labels
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import pickle
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
df = pd.read_excel("Concrete_Data.xls")

######print(df)
######print(df.describe())
######df.plot()
######plt.show()
RANDOM_SEED = 10

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def get_data():
    """ Read the iris data set and split them into training and test sets """
##    sns.pairplot(df)
##    plt.show()

    dataset = df.values
    # split into input (X) and output (Y) variables
    data = dataset[:,0:8]
    d = np.array([[0.0]*len(data[0])]*len(data))
    print(data.shape,d.shape)
##    print(data)
    for i in range(0,(len(data[0]))):
        for j in range(0,len(data)):
            maxi = np.amax(data[:,i])
            d[j,i] = data[j,i] / maxi
            
    target = dataset[:,8]
##    print(target)
    indexes_y = np.array([[0]]*len(target))
    for i in range(0,len(target)):
        indexes_y[i,0] = target[i]
    # Prepend the column of 1s for bias
    N, M  = d.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = d

    # Convert into one-hot vectors
##    num_labels = len(np.unique(indexes_y))

##    all_Y = np.eye(num_labels)[indexes_y]  # One liner trick!
    
    return train_test_split(all_X, indexes_y, test_size=0.3, random_state=RANDOM_SEED)

train_x,test_x,train_y,test_y = get_data()
print(train_x.shape,test_x.shape,train_y.shape,test_y.shape)
n_nodes_hl1 = 8
n_nodes_hl2 = 8
n_nodes_hl3 = 8

n_classes = 1
##sizes = len(train_x)
##print(sizes)
##batch_size = 100
hm_epochs = 20000

l = tf.placeholder('float')
x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}
##output_layer['weight'] *= 0.0005

# Nothing changes
def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
##    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
##    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
##    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weight']),output_layer['bias'])
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    
    cost = tf.losses.mean_squared_error(y,prediction)
    
##    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
    # Computing the gradient of cost with respect to W and b
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=l).minimize(cost)

    errors = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for epoch in range(hm_epochs):
            epoch_loss = 0

##            p2 = sess.run(prediction, feed_dict={x: train_x,y: train_y} )
##            print(p2)
            lr = 0.1
            pred = sess.run(prediction, feed_dict={x: train_x} )
            c = sess.run(cost, feed_dict={x: train_x,y: train_y} )
            sess.run(optimizer, feed_dict={x: train_x,y: train_y,l: lr} )

            rmse_val = rmse(pred, train_y)
            print("rms error is: " + str(rmse_val))
            errors.append(rmse_val)
            
##            epoch_loss = c

##            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

           
            
        p =  sess.run(prediction, feed_dict={x: test_x} )
        c = sess.run(cost, feed_dict={x: test_x,y: test_y} )
        print('Loss For testing Dataset:',c)

        rmse_val = rmse(p, test_y)
        print("rms error is: " + str(rmse_val))

        plt.scatter(test_y, p)
        plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.show()
    
        plt.plot(errors)
        plt.show()

train_neural_network(x)
