import numpy as np

def encode_val(arr,lower,upper,size):
	ori_shape = arr.shape
	arr = arr.flatten()
	new_arr = np.zeros(arr.shape)
	for i in range(arr.shape[0]):
		res = (arr[i]-lower)%size
		if res<size/2:
			new_arr[i] = int((arr[i]-lower)/size)
		else:
			new_arr[i] = int((arr[i]-lower)/size)+1
	return new_arr.reshape(ori_shape).astype(np.int32)

def decode_val(arr,lower,upper,size):
	ori_shape = arr.shape
	arr = arr.flatten()
	new_arr = np.zeros(arr.shape)
	for i in range(arr.shape[0]):
		new_arr[i] = arr[i]*size+lower
	return new_arr.reshape(ori_shape)

def val_to_one_hot(arr,class_num):
	new_arr = np.zeros((len(arr),class_num)).astype(np.int32)
	for i in range(len(arr)):
		# print(arr[i])
		new_arr[arr[i]] = 1
	return new_arr
def one_hot_to_val(arr):
	new_arr = np.zeros((len(arr),))
	for i in range(len(arr)):
		for j in range(len(arr[i])):
			if arr[i][j]==1:
				new_arr[i] = j
			break
	return new_arr




if __name__=="__main__":
	f0 = np.loadtxt("./train_data_f0")[:,0:10]
	mean = f0.mean(axis=1)
	enc_mean = encode_val(mean,0,400,20)
	dec_mean = decode_val(enc_mean,0,400,20)
	print(np.sqrt(np.square(mean-dec_mean).mean()))