import numpy as np
from sets import Set

def rmse(a,b):
	return np.sqrt(np.square(a-b).mean(axis=1)).mean()
def normalize(arr):
	mean = arr.mean(axis=1).reshape((arr.shape[0],1))
	std = (arr.std(axis=1)+0.0000001).reshape((arr.shape[0],1))
	shape = (arr-mean)/std
	return shape,mean,std


if __name__=="__main__":
	true_f0_file = "../../seq_op/my_data/train_test_data/test_data/test_f0"
	predict = "./decode_predict"
	a = np.loadtxt(true_f0_file)
	b = np.loadtxt(predict)
	print(rmse(a,b))
	print("abs error:")
	print(np.abs(a-b).mean(axis=0))
	