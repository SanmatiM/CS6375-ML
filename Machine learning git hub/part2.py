import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Class for training and evaluating a Linear Regression model
class LinearRegressionTrainer:
    def __init__(self, data_url):
        # Load the dataset from the given URL
        self.data = pd.read_csv(data_url)
        dropping_attributes = ['1', '2', '3', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
        self.data.drop(dropping_attributes, axis=1, inplace=True)
        # Extract features (X) and target (y)
        self.X = self.data.iloc[:, 1:-1]  # Exclude the first column (Student ID) and the last column (GRADE)
        self.y = self.data['GRADE']
        # Split the data into training and testing sets (80% train, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

    def train(self, learning_rates, iteration_values):
        best_learning_rate = None
        best_iterations = None
        best_mse = float('inf')

        mse_values = []
        iteration_numbers = []
        for lr in learning_rates:
            for num_iters in iteration_values:
                # Create the SGDRegressor model with specified learning rate and iterations
                model = SGDRegressor(max_iter=num_iters, alpha=lr)
                # Train the model
                model.fit(self.X_train, self.y_train)
                # Make predictions on the test set
                y_pred = model.predict(self.X_test)
                # Evaluate the model by calculating Mean Squared Error (MSE)
                mse = mean_squared_error(self.y_test, y_pred)

                # Store the MSE value and iteration number for the plot
                mse_values.append(mse)
                iteration_numbers.append(num_iters)

                # Check if this combination of parameters resulted in a lower MSE
                if mse < best_mse:
                    best_mse = mse
                    best_learning_rate = lr
                    best_iterations = num_iters

                # Log the parameters and MSE value in a log file
                self.log_parameters(lr, num_iters, mse)
                # Create a scatter plot to visualize the predictions
                self.plot_scatter_plot1(y_pred, lr, num_iters)
        self.plot_scatter_plot2(mse_values,iteration_numbers)
        self.plot_scatter_plot3(self.X_test,y_pred)
        # Print the optimal parameters and lowest MSE
        self.print_optimal_parameters(best_learning_rate, best_iterations, best_mse)

    def log_parameters(self, lr, num_iters, mse):
        # Log the parameters and MSE value in a log file
        with open('log_part2.txt', 'a') as log_file:
            log_file.write(f'Learning Rate: {lr}\n')
            log_file.write(f'Number of Iterations: {num_iters}\n')
            log_file.write(f'MSE on Test Data: {mse}\n')

    def plot_scatter_plot1(self, y_pred, lr, num_iters):
        # Create a scatter plot to visualize the predicted vs. actual values
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.title(f"Predicted vs. Actual Values for learning rate: {lr} and number of iterations: {num_iters}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)
        # Add a diagonal line for reference (perfect predictions)
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], linestyle='--', color='red', linewidth=2)

    @staticmethod
    def plot_scatter_plot2(mse_values,iteration_numbers):
        plt.figure(figsize=(8, 6))
        plt.plot(iteration_numbers, mse_values, marker='o', linestyle='-')
        plt.title("MSE vs. Number of Iterations")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.grid(True)
    
    @staticmethod
    def plot_scatter_plot3( X_test, y_pred):
        # Extract attributes for the 3D scatter plot
        x = X_test['17']
        y = X_test['22']
        z = y_pred
        
        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', marker='o')
        ax.set_xlabel('Weekly study hours')
        ax.set_ylabel('Attendance to classes')
        ax.set_zlabel('Predicted grade')
        plt.title('Predicted grade VS Weekly study hours and attendance to classes')

    def print_optimal_parameters(self, best_learning_rate, best_iterations, best_mse):
        # Print the optimal parameters and lowest MSE
        print(f"Optimal Learning Rate: {best_learning_rate}")
        print(f"Optimal Number of Iterations: {best_iterations}")
        print(f"Lowest MSE on Test Data: {best_mse}")

if __name__ == "__main__":
    data_url = "https://raw.githubusercontent.com/SanmatiM/CS6375-Assignment1/main/data/student_performance.csv"
    trainer = LinearRegressionTrainer(data_url)

    learning_rates = [0.0001, 0.001, 0.01, 0.004]
    iteration_values = [100, 500, 1000, 2000]

    # Train the linear regression model with different hyperparameters
    trainer.train(learning_rates, iteration_values)
    plt.show()
