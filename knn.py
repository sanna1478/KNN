"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is compute_distances, predict_labels, compute_accuracy
and find_best_k.
"""

import numpy as np
import json


###### Q 2.1 ######
def compute_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	"""
	#####################################################
	dists = np.zeros((np.shape(X)[0], np.shape(Xtrain)[0]))
	for test_idx in range(np.shape(X)[0]):
		test_row = X[test_idx]
		for train_idx in range(np.shape(Xtrain)[0]):
			train_row = Xtrain[train_idx]
			dists[test_idx][train_idx] = np.linalg.norm(test_row - train_row)

	#####################################################
	return dists


###### Q 2.2 ######
def predict_labels(k, ytrain, dists):
	"""
	Given a matrix of distances between test points and training points,
	predict a label for each test point.
	Inputs:
	- k: The number of nearest neighbors used for prediction.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  gives the distance betwen the ith test point and the jth training point.
	Returns:
	- ypred: A numpy array of shape (num_test,) containing predicted labels for the
	  test data, where y[i] is the predicted label for the test point X[i].
	"""
	#####################################################
	ypred = np.zeros(np.shape(dists)[0])
	for test_idx in range(np.shape(dists)[0]):
		test = dists[test_idx]
		near_neigh = sorted(range(len(test)), key=lambda k: test[k])
		count = dict()
		for idx in near_neigh[:k+1]:
			if ytrain[idx] in count:
				count[ytrain[idx]] += 1
			else:
				count[ytrain[idx]] = 1
		val = list(count.values())
		labels = list(count.keys())
		ypred[test_idx] = labels[val.index(max(val))]
	#####################################################
	return ypred


###### Q 2.3 ######
def compute_accuracy(y, ypred):
	"""
	Compute the accuracy of prediction based on the true labels.
	Inputs:
	- y: A numpy array with of shape (num_test,) where y[i] is the true label
	  of the ith test point.
	- ypred: A numpy array with of shape (num_test,) where ypred[i] is the
	  prediction of the ith test point.
	Returns:
	- acc: The accuracy of prediction (scalar).
	"""
	#####################################################
	# equal = np.sum(y == ypred, where=True)
	# acc = equal/len(ypred)
	cnt = 0
	for i in range(len(ypred)):
		if y[i] == ypred[i]:
			cnt+=1
	acc = float(cnt)/len(ypred)
	return acc
	#####################################################
	return acc


###### Q 2.4 ######
def find_best_k(K, ytrain, dists, yval):
	"""
	Find best k according to validation accuracy.
	Inputs:
	- K: A list of ks.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	- yval: A numpy array with of shape (num_val,) where y[i] is the true label
	  of the ith validation point.
	Returns:
	- best_k: The k with the highest validation accuracy.
	- validation_accuracy: A list of accuracies of different ks in K.
	"""

	#####################################################
	validation_accuracy = []
	acc = None
	for k in K:
		ypred = predict_labels(k, ytrain, dists)
		acc = compute_accuracy(yval, ypred)
		validation_accuracy.append(acc)
		print("The validation accuracy is",acc,"when k=",k)
	high_acc = max(validation_accuracy)
	best_k = K[validation_accuracy.index(high_acc)]
	print("The best_k:",best_k)
	#####################################################
	return best_k, validation_accuracy


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_processing(data):
	train_set, valid_set, test_set = data['train'], data['valid'], data['test']
	Xtrain = train_set[0]
	ytrain = train_set[1]
	Xval = valid_set[0]
	yval = valid_set[1]
	Xtest = test_set[0]
	ytest = test_set[1]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)

	return Xtrain, ytrain, Xval, yval, Xtest, ytest

def main():
	input_file = 'mnist_subset.json'
	output_file = 'knn_output.txt'

	with open(input_file) as json_data:
		data = json.load(json_data)

	#==================Compute distance matrix=======================
	K = [1, 3, 5, 7, 9]

	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	dists = compute_distances(Xtrain, Xval)
	from pprint import pprint
	print("My Solution KBB dist matrix:")
	pprint(dists)

	#===============Compute validation accuracy when k=5=============
	k = 5
	ypred = predict_labels(k, ytrain, dists)
	acc = compute_accuracy(yval, ypred)
	# print("The validation accuracy is", acc, "when k =", k)

	#==========select the best k by using validation set==============
	best_k,validation_accuracy = find_best_k(K, ytrain, dists, yval)


	#===============test the performance with your best k=============
	dists = compute_distances(Xtrain, Xtest)
	ypred = predict_labels(best_k, ytrain, dists)
	test_accuracy = compute_accuracy(ytest, ypred)

	#====================write your results to file===================
	f=open(output_file, 'w')
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], validation_accuracy[i])+'\n')
	f.write('%s %.3f' % ('test', test_accuracy))
	f.close()

if __name__ == "__main__":
	main()
