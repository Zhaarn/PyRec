from math import sqrt
from cvxopt import matrix

class Recsys:

    def __init__(self, user_matrix, item_matrix, mean=None, user_bias=None, item_bias=None):
        if not (user_matrix or item_matrix):
            print "Please enter valid parameters"
        self._U = user_matrix
        self._I = item_matrix
        self._b = (user_bias, item_bias)

    def predict(self, user, item):
        print "Prediction: %f" %  self._mu + self._b[0][user] - self._b[1][item] - (self._U[user, :] * self._I[:, item])[0]
    
    def similar(self, item1, item2):
        print "Similarity: %f" % (self._I[:, item1].trans() * self._I[:, item2])[0]
        
