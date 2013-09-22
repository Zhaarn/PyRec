from Recsys import Recsys
from Algorithm import Algorithm
from DataModel import DataModel

alg = Algorithm()
dat = DataModel()
dat.load_from_file("../u.data")
dat.split_train_validate_test()
training_data = dat.get_training_data()
validation_data = dat.get_validation_data()
test_data = dat.get_test_data()
alg.train(training_data, dat.get_size())
alg.rmse(validation_data, dat.get_mean(), dat.get_biases())
alg.rmse(test_data, dat.get_mean(), dat.get_biases())
rec = Recsys(alg.get_user_matrix(), alg.get_item_matrix(), dat.get_mean(), *dat.get_biases())
rec.predict(5, 600)
rec.similar(5, 600)

