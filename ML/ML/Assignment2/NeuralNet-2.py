#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import keras
import tensorflow as tf


class NeuralNet:

    def read_file(self, id):
        url = 'https://drive.google.com/uc?export=download&id=' + id
        headers = { 'Accept': 'application/text' }
        r = requests.get(url, headers = headers)
        with open('data.csv', "w") as f:
            f.write(r.text)

    def __init__(self, dataFile, header=True):
        self.read_file('1CHo_HhlEn30OZMit5_QMntxV0D-nrm2S')
        self.raw_input = pd.read_csv(dataFile, encoding='utf-8')





    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        plt.figure(figsize=(16, 6))
        #   Store heatmap object in a variable to easily access it when you want to include more features (such as title).
        # Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
        heatmap = sns.heatmap(self.raw_input.corr(), vmin=-1, vmax=1, annot=True)
        print(heatmap)
        self.processed_data = self.raw_input
        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)
        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['sigmoid', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]
        table=[]
        table.append(["Activation Func", "Learning Rate", "No, of epochs", "No. of Hidden Layers", "R2 Score"])
        for activation in activations:
            for l_rate in learning_rate:
                for epoch in max_iterations:
                    for layers in num_hidden_layers:
                        table_row=[]
                        table_row.append(activation)
                        table_row.append(l_rate)
                        table_row.append(epoch)
                        table_row.append(layers)
                        model = Sequential()
                        model.add(Dense(12, input_dim=6, activation=activation))
                        for i in range(1,layers):
                            model.add(Dense(12, activation=activation))
                        model.add(Dense(3, activation="softmax"))
                        optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
                        model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=["accuracy"])
                        model.fit(X_train, y_train, epochs=epoch)
                        table_row.append(model.score(X_test, y_test))
                        table.append(table_row)

        print(tabulate(table, headers='firstrow'))
        

                        




        # Create the neural network and be sure to keep track of the performance
        #   metrics

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("data.csv") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
