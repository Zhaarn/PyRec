from math import sqrt
from cvxopt import matrix

class Recsys:

    def __init__(self, user_matrix, item_matrix, mean=None, user_bias=None, item_bias=None):
        """
        Initializes the recommender system base on a trained model

        :param user_matrix: a matrix containing vectors of feature values for each user
        :type user_matrix: cvxopt matrix
        :param item_matrix: a matrix containing vectors of feature values for each item
        :type item_matrix: cvxopt matrix:
        :param mean: the mean of the training set
        :type mean: float
        :param user_bias: a list containing the per user biases
        :type user_bias: list
        :param item_bias: a list containing the per item biases
        :type item_bias: list
        """
        if not (user_matrix or item_matrix):
            print "Please enter valid parameters"
        self._U = user_matrix
        self._I = item_matrix
        self._b = (user_bias, item_bias)

    def predict(self, user, item):
        """
        Predicts the rating a given user would give to an item

        :param user: user index
        :type user: int
        :param item: item index:
        :type item: int
        """
        print "Prediction: %f" %  self._mu + self._b[0][user] - self._b[1][item] - (self._U[user, :] * self._I[:, item])[0]
    
    def similar(self, item1, item2):
        """
        Gives the similarity between two items/users
        
        :param item1: item/user index
        :type user: int
        :param item2: item/user index:
        :type item: int
        """
        
        print "Similarity: %f" % (self._I[:, item1].trans() * self._I[:, item2])[0]
        
