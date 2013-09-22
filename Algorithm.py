from cvxopt import matrix, spmatrix, uniform, mul
from math import sqrt
import matplotlib.pyplot as plt

class Algorithm:
    
    def _create_matrices(self, training_data, size, k=30):
        """
        Converts the training set into matrix form
        
        :param training_data: a dataset
        :type training_data: a tuple of lists
        :param size: dimensions of the matrix
        :type size: tuple
        :param k: the size of the common feature space for users and items
        :type k: int
        """
        print "Creating matrices.......\n"
        self._M = spmatrix(training_data[2], training_data[0], training_data[1], size)
        rated = [1]*len(training_data[2])
        self._R = spmatrix(rated, training_data[0], training_data[1], size)
        self._U = uniform(size[0], k, 0, 0.2)
        self._I = uniform(k, size[1], 0, 0.2)
    
    def get_user_matrix(self):
        """
        Returns a matrix containing feature vectors for each user
        """
        if self._U is not None:
            return self._U
        else:
            print "The user matrix has not been initialized"

    def get_item_matrix(self):
        """
        Returns a matrix containing feature vectors for each item 
        """
        if self._I is not None:
            return self._I
        else:
            print "The item matrix has not been initialized"

    def get_training_matrix(self):
        """
        Returns a matrix containing the training set
        """
        if self._M is not None:
            return self._M
        else:
            print "The training matrix has not been initialized"

    #TODO: Add cost printing, make constants for plot measurements
    def _gradient_descent(self, cost_function=None, gradient=None, alpha=.001, reg=.00001, plot_cost=False):
        """
        Performs gradient descent on the training matrix.

        :param cost_function: a cost function
        :type cost_function: function
        :param gradient: the gradient of the cost_function
        :type gradient: function
        :param alpha: the learning rate
        :type alpha: float
        :param reg: the regularlization parameter
        :type reg: float:
        :param plot_cost: whether to plot the cost function
        :type plot_cost: boolean
        """
        print "Performing gradient descent.......\n"
        
        if cost_function is None:
            cost_function=self._cost_function
        if gradient is None:
            gradient = self._gradient

        if plot_cost:
            x_data = []
            y_data = []
        
        #print cost_function(reg)
        for i in range(0,2000):
            grad_U, grad_I = gradient(reg)
            self._U -= alpha*grad_U
            self._I -= alpha*grad_I
            if plot_cost:
                cost = cost_function(reg)
                y_data.append(cost)
                x_data.append(i)
        
        if plot_cost:
            plt.scatter(x_data, y_data)
            plt.xlim((-10,2000))
            plt.ylim((0,1))
            plt.show()

    def _gradient(self, reg, M=None, U=None, I=None, R=None):
        """
        Computes the value of the gradient

        :param reg: the regularization parameter
        :type reg: float
        :param M: the training matrix
        :type M: cvxopt spmatrix
        :param U: the user matrix
        :type U: cvxopt matrix
        :param I: the item matrix
        :type I: cvxopt matrix
        :param R: a matrix containing a 1 where a value is present in the training matrix
        :type R: cvxopt spmatrix
        """
        if M is None:
            M = self._M
        if U is None:
            U = self._U
        if I is None:
            I = self._I
        if R is None:
            R = self._R
            
        print "Calculating gradient.......\n"
        grad_U = mul((U * I - M), R) * I.trans() + reg * U
        grad_I = (mul((U * I - M), R).trans() * U).trans() + reg * I
        return (grad_U, grad_I)

    def _cost_function(self, reg, M=None, U=None, I=None, R=None):
        """
        Computes the value of the cost function

        :param reg: the regularization parameter
        :type reg: float
        :param M: the training matrix
        :type M: cvxopt spmatrix
        :param U: the user matrix
        :type U: cvxopt matrix
        :param I: the item matrix
        :type I: cvxopt matrix
        :param R: a matrix containing a 1 where a value is present in the training matrix
        :type R: cvxopt spmatrix
        """
        if M is None:
            M = self._M
        if U is None:
            U = self._U
        if I is None:
            I = self._I
        if R is None:
            R = self._R

        mu = self._mean
        #print "Calculating cost.........\n"
        error = mul((M - U * I), R) 
        sqerror = mul(error, error)
        cost = sum(sqerror) + reg * (sum(U**2) + sum(I**2))
        return (0.5*cost)/(self._ltrain)

    def train(self, training_data, size):
        """
        Trains the dataset

        :param training_data: the training set
        :type training_data: a 3-tuple of lists
        :param size: the dimensions of the training matrix
        :type size: tuple
        """
        print "Training.......\n"

        if training_data is None:
            print "Please supply valid training data"

        self._create_matrices(training_data, size)
        self._gradient_descent()


    def rmse(self, data, mean=None, biases=None):
        """
        Calculates the rmse of the model on a given data set
        
        :param data: the test data
        :type data: 3-tuple of lists
        :param mean: the mean of the training set
        :type mean: float
        :param biases: the biases of the training set
        :type biases: a 2-tuple of float lists
        """
        if data is None:
            print "Please supply valid test data"

        error = self._test(data, mean, biases)
        print "Test error: %f" % (error) 
        
    
    def _test(self, test_data, mean=None, biases=None ):
        """
        Calculates the rmse of the model on a given data set
        
        :param test_data: the test data
        :type test_data: 3-tuple of lists
        :param mean: the mean of the training set
        :type mean: float
        :param biases: the biases of the training set
        :type biases: a 2-tuple of float lists
        """
        data = zip(test_data[0], test_data[1], test_data[2])
        U = self._U
        I = self._I
        mu = mean
        b = biases
        ltest = float(len(data))
        cost = 0
        for r, c, v in data:
            if mean and biases:
                cost += (v - mu - b[0][r] - b[1][c] - (U[r, :] * I[:, c])[0])**2
            else:
                cost += (v  - (U[r, :] * I[:, c])[0])**2
        return sqrt(cost/ltest)


