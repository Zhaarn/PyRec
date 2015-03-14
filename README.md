PyRec
=====

Python Recommender System with SVD using gradient descent. The model can be trained using any data set that contains users, and items, where one or more users have rated at least one item.  For testing this recommender system, I used the Movielens dataset (http://www.grouplens.org/node/73). 


Dependencies
=====

This recommender system implementation requires Python 2.7+, numpy (www.numpy.org) and the CVXOPT library (www.cvxopt.org). 

Example
=====

Here is some example code for the project:

```python

from recsys import Recsys
from algorithm import Algorithm
from data_model import DataModel

# intialize classes
alg = Algorithm()
dat = DataModel()

# load file
dat.load_from_file("data.csv")

# split data set
dat.split_train_validate_test()

# get training, validation and test sets 
training_data = dat.get_training_data()
validation_data = dat.get_validation_data()
test_data = dat.get_test_data()

# train data
alg.train(training_data, dat.get_size())

#calculate errors
alg.rmse(validation_data, dat.get_mean(), dat.get_biases())
alg.rmse(test_data, dat.get_mean(), dat.get_biases())

# run recommender system
rec = Recsys(alg.get_user_matrix(), alg.get_item_matrix(), dat.get_mean(), *dat.get_biases())
rec.predict(5, 600) # predict the user #5's rating for item #600
rec.similar(5, 10) # prints out the similarity between items #5 and #10
```


