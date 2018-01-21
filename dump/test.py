import numpy as np

def rmse(a,b):
	return np.sqrt(np.square(a-b).mean(axis=1)).mean()

if __name__=="__main__":
	a = np.loadtxt("../phrase_lstm_prediction",delimiter=" ")
	a = a.mean(axis=1).reshape((a.shape[0],1))

	b = np.loadtxt("../../mandarine/gen_f0/train_dev_data_vector/dev_data_f0_vector",delimiter=" ")
	b = b.mean(axis=1).reshape((b.shape[0],1))

	print(rmse(a,b))

