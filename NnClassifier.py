"""
This module applies a basic neural network (1 hidden layer and 1 output layer, this is 
from the XOR problem) to the compass problem where the features are created manually. 
"""

# Import the libraries
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import the dataset
# NOTE: Angles are taken counter-clockwise, similar to the unit circle
dataset = pd.read_csv('/home/matt/Documents/PossiblePortfolio/NeuralNetwork_Compass/FeatureVector.csv')

X = dataset.iloc[:, 1:3].values # Feature matrix
y = dataset.iloc[:, 3].values # Labels

# Center and normalize features values around origin (prior possible values: 0, 90, 180, 270) 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X = sc.fit_transform(X) 

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
y_train = np.array(y_train).reshape(y_train.shape[0],1)
y_test = np.array(y_test).reshape(y_test.shape[0],1)

# Convert training data to lists so we can shuffle the order using random.shuffle
X_train = X_train.tolist()
y_train = y_train.tolist()

# Shuffle the training X and y 
combined = list(zip(X_train, y_train)) 
random.shuffle(combined)
X_train[:], y_train[:] = zip(*combined)

# Convert data back to numpy arrays for further processing
X_train = np.array(X_train)
y_train = np.array(y_train)

# Separate the training points into ones and zeros
ones = X_train[np.where(y_train == 1)[0]]
zeros = X_train[np.where(y_train == 0)[0]]

# Plot the training points before we start training the NN 
plt.figure()
plt.plot(ones[:, 0], ones[:, 1], "ro", label="1")
plt.plot(zeros[:, 0], zeros[:, 1], "bo", label="0")
plt.legend(loc="best")
plt.show()

####### NEURAL NETWORK
import tensorflow as tf

def plotGrid(session, hypo,  numTrainPts):
    """ Plots the training points and the hyperplane created by the NN """ 
    # Domain to be displayed
    Xs = np.linspace(-2,2)
    Ys = np.linspace(-2,2)

    # Allocate space for the points that fall in the positive class
    Xs_p, Ys_p = [],[]
    
    # Go through every point in the domain
    for x in Xs:
        for y in Ys:
            # Calculate the class of the point and append the point to the list if it's in the positive class
            plot_input = [[x,y]] * numTrainPts
            output = session.run(hypo,feed_dict={x_: plot_input})
            if output[0] > 0.5:
                Xs_p.append(x)
                Ys_p.append(y)
    
    # Plot the 
    plt.plot(Xs_p, Ys_p, "y.")
    plt.plot(ones[:, 0], ones[:, 1], "ro", label="1")
    plt.plot(zeros[:, 0], zeros[:, 1], "bo", label="0")
    plt.show()

badModel = True 
costThresh = 0.05 # Desired cost threshold of the NN. This is empirically determined as the absolute minimum
while badModel:
    # Setting up model
    x_ = tf.placeholder(tf.float32, shape=X_train.shape, name="x-input") # Input feature vector: {x1,x2}
    y_ = tf.placeholder(tf.float32, shape=y_train.shape, name="y-input") # Output: y1

    # First layer
    Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="Theta1")  
    Bias1 = tf.Variable(tf.zeros([2]), name="Bias1")
    A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)
    
    # Second/Output layer
    Theta2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="Theta2")
    Bias2 = tf.Variable(tf.zeros([1]), name="Bias2")
    Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

    # cross-entropy loss
    cost = tf.reduce_mean(((y_ * tf.log(Hypothesis)) + ((1 - y_) * tf.log(1.0 - Hypothesis))) * -1) 

    # Optimizer
    train_step = tf.train.MomentumOptimizer(0.01,0.9).minimize(cost)

    # Initialize the model
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Train the model
    for i in range(200000):
        sess.run(train_step, feed_dict={x_: X_train, y_: y_train})

        # Report the cost every 10000 epochs
        if i % 10000 == 0:
            print('Epoch ', i)
            curr_cost = sess.run(cost, feed_dict={x_: X_train, y_: y_train})
            print('cost ', curr_cost)
            # End training if the cost reaches desired value
            if curr_cost < costThresh:
                badModel = False
                break

    # Calculate the NN output of the test points
    testOutput = []
    for testPoint in X_test:
        test_in = [testPoint] * y_train.shape[0]
        test_out = sess.run(Hypothesis,feed_dict={x_: test_in})
        if test_out[0] < 0.5:
            testOutput.append(0)
        else:
            testOutput.append(1)

    # Report the NN output of the test points next to the actual
    print('Calculated ', np.array(testOutput))
    print('Actual ', y_test)

    # Plot the NN hyperplane along with the training points
    if not badModel: 
        plotGrid(sess, Hypothesis, y_train.shape[0])
    
