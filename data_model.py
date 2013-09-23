from collections import defaultdict
import sys
import codecs
from random import shuffle
from operator import itemgetter

class DataModel:
    

    def load_from_file(self, filename, sep ='\t', format={'value':2, 'row':0, 'col':1}):
        """
        Loads data from a delimited file.  The file must have columns corresponding to
        user, item and rating field.  Most of the code in this method was borrow from another
        python recommender system at (https://github.com/ocelma/python-recsys.git)
        
        :param filename: The name of the data file
        :type filename: string
        :param sep: The column delimiter of the file
        :type sep: string
        :param format: The format of the file contents, where the label of a column acts as a
            key and the index of the column is the value
        :type format: dict
        """
        rows = []
        columns = []
        values = []
        sys.stdout.write('Loading %s\n' % filename)

        i = 0
        for line in codecs.open(filename, 'r', 'utf8'):
            data = line.strip('\r\n').split(sep)
            value = None
            if not data:
                raise TypeError('Data is empty or None!')
            if not format:
                try:
                    value, row_id, col_id = data
                except:
                    value = 1
                    row_id, col_id = data
            else:
                try:
                     # Default value is 1
                    try:
                        value = data[format['value']]
                    except KeyError, ValueError:
                        value = 1
                    try: 
                        row_id = data[format['row']]
                    except KeyError:
                        row_id = data[1]
                    try:
                        col_id = data[format['col']]
                    except KeyError:
                        col_id = data[2]
                    row_id = row_id.strip()
                    col_id = col_id.strip()
                    if format.has_key('ids') and (format['ids'] == int or format['ids'] == 'int'):
                        try:
                            row_id = int(row_id)
                        except:
                            print 'Error (ID is not int) while reading: %s' % data #Just ignore that line
                            continue
                        try:
                            col_id = int(col_id)
                        except:
                            print 'Error (ID is not int) while reading: %s' % data #Just ignore that line
                            continue
                except IndexError:
                    #raise IndexError('while reading %s' % data)
                    print 'Error while reading: %s' % data #Just ignore that line
                    continue
            # Try to convert ids to int
            try:
                row_id = int(row_id)
            except: pass
            try:
                col_id = int(col_id)
            except: pass
            try:
                value = float(value)
            except: pass
            # Add to list
            try:
                rows.append(row_id-1)
                columns.append(col_id-1)
                values.append(value)
            except:
                sys.stdout.write('\nError while reading (%s, %s, %s). Skipping this tuple\n' % (value, row_id, col_id))
            i += 1

        self._rows = rows
        self._columns = columns
        self._values = values
        self._data = (rows, columns, values)
        self._size = len(values)

    def get_mean(self):
        """
        Returns the mean of the training data set if it has been
        mean-normalized
        """
        if self._mean is not None:
            return self._mean
        else:
            print "The data has not been mean-normalized"
    
    def get_biases(self):
        """
        Returns a tuple of lists containing the values of the per item and
        per user biases if the biased have been removed from the training
        set
        """
        if self._biases is not None:
            return self._biases
        else:
            print "The data still has user and item biases"
     
    def get_rows(self):
        """
        Returns a list containing the row indices of each element in the dataset
        """
        if self._rows is not None:
            return self._rows
        else:
            print "No elements have been added to the dataset"
    
    def get_columns(self):
        """
        Returns a list containing the column indices of each element in dataset
        """
        if self._columns is not None:
            return self._columns
        else:
            print "No elements have been added to the dataset"

    def get_values(self):
        """
        Returns the values containing the rating values in the dataset
        """
        if self._values is not None:
            return self._values
        else:
            print "No elements have been added to the dataset"

    def get_training_data(self):
        """
        Returns a 3-tuple containing 3 lists (rows, columns, values) that comprise
        the data in the training set
        """
        if self._train is not None:
            return self._train
        else:
            print "There is currently no training data"

    def get_test_data(self):
        """
        Returns a 3-tuple containing 3 lists (rows, columns, values) that comprise
        the data in the test set
        """
        if self._test is not None:
            return self._test
        else:
            print "There is currently no test data"

    def get_validation_data(self):
        """
        Returns a 3-tuple containing 3 lists (rows, columns, values) that comprise
        the data in the validation set
        """
        if self._validate is not None:
            return self._validate
        else:
            print "There is currently no validation data"

    def get_size(self):
        """
        Returns the dimensions of the matrix
        """
        if self._matrix_size is not None:
            return self._matrix_size
        else:
            print "There is no data available"

    def split_train_test(self, training_percentage=.80, normalize=True, remove_bias=True, randomize=False):
        """
        Splits the data into a training set and a test set
        
        :param training_percentage: the percentage of the dataset that is in the training set
        :type training_percentage: float
        :param normalize: whether the training set should be normalized
        :type normalize: boolean
        :param remove_bias: whether the item and user biases should be removed from the training set
        :type remove_bias: boolean
        :param randomize: whether the dataset should be randomized
        :type randomize: boolean
        """
        print "Splitting....\n"
        if self._data is None:
            "There is no data to split"
            return
        
        if randomize:
            randomized_data = self.randomize_data()
            self._rows = randomized_data[0]
            self._columns = randomized_data[1]
            self._values = randomized_data[2]

        num_rows = max(self._rows) + 1
        num_cols = max(self._columns) + 1
        self._matrix_size = (num_rows, num_cols)
        train = int(training_percentage*self._size)
        training_values = self._values[:train]
        if normalize:
            training_values = self._mean_normalize(training_values)
        if remove_bias:
            biased_data = zip(self._rows[:train], self._columns[:train], training_values)
            training_values = self._calculate_bias(bias_data, tuple(self._matrix_size))
        
        self._train = (self._rows[:train], self._columns[:train], training_values)
        self._test = (self._rows[train:], self._columns[train:], self._values[train:])

    def split_train_validate_test(self, training_percentage=.80, normalize=True, remove_bias=True, randomize=False):
        """
        Splits the data set into a training set, test set and validation set. The test and validation sets each
        contain half of the data left over from removing the training set
        
        :param training_percentage: the percentage of the dataset that is in the training set
        :type training_percentage: float
        :param normalize: whether the training set should be normalized
        :type normalize: boolean
        :param remove_bias: whether the item and user biases should be removed from the training set
        :type remove_bias: boolean
        :param randomize: whether the dataset should be randomized
        :type randomize: boolean
        """
        print "Splitting....\n"
        if self._data is None:
            print "There is no data in the data model"
            return

        if randomize:
            randomized_data = self.randomize_data()
            self._rows = randomized_data[0]
            self._columns = randomized_data[1]
            self._values = randomized_data[2]

        train = int(training_percentage*self._size)
        num_rows = max(self._rows) + 1
        num_cols = max(self._columns) + 1
        self._matrix_size = (num_rows, num_cols)
        validate = train + int((self._size - train)/2.0)
        if randomize:
            self._data = self.randomize_data(self._data)
        training_values = self._values[:train]
        if normalize:
            training_values = self._mean_normalize(training_values)
        if remove_bias:
            biased_data = zip(self._rows[:train], self._columns[:train], training_values)
            training_values = self._calculate_bias(biased_data, num_rows, num_cols)

        self._train = (self._rows[:train], self._columns[:train], training_values)
        self._validate = (self._rows[train:validate], self._columns[train:validate], self._values[train:validate])
        self._test = (self._rows[validate:], self._columns[validate:], self._values[validate:])
    
    def _randomize_data(self):
        """
        Randomizes the order of the elements in the dataset
        """
        list1_shuf = []
        list2_shuf = []
        list3_shuf = []
        index_shuf = range(len(self._values))
        shuffle(index_shuf)
        for i in index_shuf:
            list1_shuf.append(self._rows[i])
            list2_shuf.append(self._columns[i])
            list3_shuf.append(self._values[i])
        return(list1_shuf, list2_shuf, list3_shuf)
    
    def _mean_normalize(self, data):
        """
        Mean-normalizes the data in the data set

        :param data: the data to normalize
        :type data: list of values
        """
        if data is None:
            print "There is no data to normalize"
            return

        self._mean = float(sum(data))/float(len(data))
        normalized_data = [x - self._mean for x in data]
        return normalized_data
 
    def _calculate_bias(self, data, num_rows, num_cols):
        """
        Removes the user and item biases from the data

        :param data: the dataset to remove the bias from
        :type data: list of tuples
        :param num_rows: the number of rows in the dataset
        :type num_rows: int
        :param num_cols: the number of columns in the dataset
        :type num_cols: int

        """
        if data is None:
            print "There is no data to normalize"
            return
        
        user_bias = [0] * num_rows
        num_users = defaultdict(int)
        item_bias = [0] * num_cols
        num_items = defaultdict(int)

        for r, c, v in data:
            user_bias[r] += v
            num_users[r] += 1
            item_bias[c] += v
            num_items[c] += 1

        user_bias = [ x/num_users[r]  if num_users[r] != 0 else 0 for r, x in enumerate(user_bias) ]
        item_bias = [ x/num_items[c] if num_items[c] != 0 else 0 for c, x in enumerate(item_bias) ]
        self._biases = (user_bias, item_bias)
        unbiased_data = [v - user_bias[r] - item_bias[c] for r, c, v in data]
        return unbiased_data


