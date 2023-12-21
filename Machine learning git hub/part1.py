import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Class for data preprocessing
class DataPreprocessor:
    def __init__(self, data_url):
        # Initialize by reading the CSV data from the provided URL
        self.df = pd.read_csv(data_url)

    def preprocess_data(self):
        # Check for null data
        if self.df.isnull().sum().sum() == 0:
            print("Null data is absent")
        else:
            print("Null data is present")
            # Remove rows with null values
            self.df.dropna(inplace=True)

        # Check for redundant rows
        if self.df.duplicated().any():
            print("Redundant rows present")
            # Remove duplicate rows
            self.df.drop_duplicates(inplace=True)
        else:
            print("Redundant rows absent")

        # Drop the specified attributes by column numbers
        dropping_attributes = ['1', '2', '3', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
        self.df.drop(dropping_attributes, axis=1, inplace=True)

# Class for Linear Regression modeling
class LinearRegressionModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    @staticmethod
    def hypothesis(X, theta):
        return np.dot(X, theta)

    @staticmethod
    def cost_function(X, y, theta):
        m = len(y)
        predictions = LinearRegressionModel.hypothesis(X, theta)
        error = predictions - y
        return (1 / (2 * m)) * np.sum(error ** 2)

    @staticmethod
    def gradient_descent(X, y, theta, alpha, num_iterations):
        m = len(y)
        cost_history = []
        for _ in range(num_iterations):
            predictions = LinearRegressionModel.hypothesis(X, theta)
            error = predictions - y
            gradient = (1 / m) * np.dot(X.T, error)
            theta -= alpha * gradient
            cost = LinearRegressionModel.cost_function(X, y, theta)
            cost_history.append(cost)
        return theta, cost_history

    def train_and_evaluate(self, learning_rates, iteration_values):
        best_learning_rate = None
        best_iterations = None
        best_mse = float('inf')
        # Initialize lists to store MSE values and iteration numbers
        mse_values = []
        iteration_numbers = []

        for lr in learning_rates:
            for num_iters in iteration_values:
                theta = np.zeros(self.X_train.shape[1])
                theta, cost_history = self.gradient_descent(self.X_train, self.y_train, theta, lr, num_iters)

                training_error = self.cost_function(self.X_train, self.y_train, theta)

                y_pred = self.hypothesis(self.X_test, theta)
                mse_value = self.cost_function(self.X_test, self.y_test, theta)

                if mse_value < best_mse:
                    best_mse = mse_value
                    best_learning_rate = lr
                    best_iterations = num_iters

                self.log_parameters(lr, num_iters, mse_value, training_error)
                self.plot_scatter_plot1(self.y_test, y_pred, lr, num_iters)

                # Store the MSE value and iteration number for the plot
                mse_values.append(mse_value)
                iteration_numbers.append(num_iters)
        self.plot_scatter_plot2(mse_values,iteration_numbers)
        self.plot_scatter_plot3(self.X_test,y_pred)
        self.print_optimal_parameters(best_learning_rate, best_iterations, best_mse)

    @staticmethod
    def log_parameters(lr, num_iters, mse_value, training_error):
        # Log the training parameters and MSE values to a file
        with open('log_part1.txt', 'a') as log_file:
            log_file.write(f'Learning Rate: {lr}\n')
            log_file.write(f'Number of Iterations: {num_iters}\n')
            log_file.write(f'MSE on Training Data: {training_error}\n')
            log_file.write(f'MSE on Test Data: {mse_value}\n')

    @staticmethod
    def plot_scatter_plot1(y_actual, y_pred, lr, num_iters):
        # Create a scatter plot of predicted vs. actual values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_actual, y_pred, alpha=0.5)
        plt.title(f"Predicted vs. Actual Values for learning rate: {lr} and number of iterations: {num_iters}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)
        plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], linestyle='--', color='red', linewidth=2)

    @staticmethod
    def plot_scatter_plot2(mse_values, iteration_numbers):
        # Create a plot of MSE vs. number of iterations
        plt.figure(figsize=(8, 6))
        plt.plot(iteration_numbers, mse_values, marker='o', linestyle='-')
        plt.title("MSE vs. Number of Iterations")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.grid(True)

    @staticmethod
    def plot_scatter_plot3(X_test, y_pred):
        # Extract attributes for the 3D scatter plot
        x = []
        y = []
        for i in X_test:
            x.append(i[1])  # Assuming index 1 corresponds to 'Weekly study hours'
            y.append(i[6])  # Assuming index 6 corresponds to 'Attendance to classes'
        z = y_pred
        
        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x, y, z, c='b', marker='o')

        ax.set_xlabel('Weekly study hours')
        ax.set_ylabel('Attendance to classes')
        ax.set_zlabel('Predicted grade')
        plt.title('Predicted grade VS Weekly study hours and attendance to classes')

    @staticmethod
    def print_optimal_parameters(best_learning_rate, best_iterations, best_mse):
        # Print the optimal parameters
        print(f"Optimal Learning Rate: {best_learning_rate}")
        print(f"Optimal Number of Iterations: {best_iterations}")
        print(f"Lowest MSE on Test Data: {best_mse}")

if __name__ == "__main__":
    data_url = "https://raw.githubusercontent.com/SanmatiM/CS6375-Assignment1/main/data/student_performance.csv"
    
    data_processor = DataPreprocessor(data_url)
    data_processor.preprocess_data()

    X = np.array(data_processor.df.drop("GRADE", axis=1))[:, 1:].astype('float64')
    y = data_processor.df["GRADE"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegressionModel(X_train, y_train, X_test, y_test)

    learning_rates = [0.0001, 0.001, 0.01, 0.004]
    iteration_values = [100, 500, 1000, 2000]

    model.train_and_evaluate(learning_rates, iteration_values)
    plt.show()
