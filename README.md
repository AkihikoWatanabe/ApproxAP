# ApproxAP
A python implementation of ApproxAP. ApproxAP is one of the listwise learning to rank approach to optimize Average Precision.
For details about ApproxAP, see the following papers:
```
A General Approximation Framework for Direct Optimization of Information Retrieval Measures, Qin+, Information Retrieval, 2010.
```

# Requirements
* python 2.7
* tqdm
* numpy
* scipy
* joblib

# Example
## Training
```python
from updater import Updater
from weight import Weight

# number of maximum epochs
epochs = 200

# number of maximum number of features
max_feature_num = 5

# learning rate 
eta = 0.01

# scaling constant for position function
alpha = 10

# scaling constant for truncation function
beta = 1

# make training data
# x_train represents feature vector using dict
#       - key: qid
#	- value: feature vectors using scipy.sparse.csr_matrix
# y_train represents relevancy labels (e.g. 5 scale ratings or binary) corresponding to each feature_vector using dict
#	- key: qid
#	- value: relevancy vectors
x_train, y_train = make_data()

weight = Weight(max_feature_num)

updater = Updater(eta=eta, alpha=alpha, beta=beta)

for _ in xrange(epochs):
	# update weight using DBGD
	updater.update(x_train, y_train, weight)
	# dump weight parameter
	weight.dump_weight("./models/approx_ap")
```

## Testing
```python
from weight import Weight
import predictor import Predictor

# make test data
x_test, y_test = make_data()

# load trained weight parameters from model file
# second argument means number of epochs for weight that you want to load
weight = Weight()
weight.load_weight("./models/approx_ap", 30)

predictor = Predictor()

# get result rankings for x_test
for qid, features in x_test.items():
	labels = y_test[qid]
	# ranking is represented as list and its element is composed of (true_label, case_id, score) by descending order of score
	ranking = predictor.predict_and_ranks(features, labels, weight.get_weight())
```
