# CS6375-Assignment1
Assignment 1: Linear Regression using Gradient Descent
Libraries required:
Install the below libraries before running the python code.

 

Numpy
Pandas
Sklearn
Matplotlib

 

Dataset : Student Performance Evaluation

https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation

 

How to run?

 

Run command :

 

python part1.py

 

After running part1 the best Mean Squared Error (MSE) value will be printed and also the 16 graphs will be plotted for different learning rates and different number of iterations.

 

Check for the terminal for best MSE with specified learning rate and number of iterations and check for the same in graph.

 

2.Python part2.py

 

In this part, we use the sklearn library to calculate mean squared error.

 

Below are the used libraries:


from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_squared_error

 

The calculated MSE will be printed on the terminal which is calculated for the same dataset used in part 1.
