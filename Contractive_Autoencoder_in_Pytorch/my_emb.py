import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import sys
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import copy

def show_heatmap(arr):
	plt.imshow(arr, cmap='hot', interpolation='nearest')
	plt.show()

class CAE(nn.Module):
	def __init__(self,win_size):
		super(CAE, self).__init__()
		self.h_size = 100
		self.win_size = win_size
		self.fc1 = nn.Linear(10,self.h_size) # Encoder
		self.fc3 = nn.Linear(self.h_size*(self.win_size-1),self.h_size) # Encoder
		self.fc2 = nn.Linear(self.h_size,self.h_size)
		self.fc4 = nn.Linear(self.h_size,10)
		# self.fc2 = nn.Linear(self.h_size,10)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()


	def forward(self, x):
		batch_size,_,_ = x.size()
		x = self.fc1(x)
		# x = torch.sum(x,dim=1)
		x = x.view(batch_size,(self.win_size-1)*self.h_size)
		x = self.relu(x)
		x = self.fc3(x)
		x = self.sigmoid(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc4(x)
		return x

def loss_function(W, x, recons_x, h,mse_loss):
	lam = 0.001
	#mse = (x-recons_x).pow(2).sum()
	mse = mse_loss(recons_x, x)
	"""
	W is shape of N_hidden x N. So, we do not need to transpose it as opposed to http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
	"""
	dh = h*(1-h) # N_batch x N_hidden
	# zero = Variable(torch.FloatTensor([0]))
	# one = Variable(torch.FloatTensor([1]))
	# dh = torch.gt(h,zero)
	# dh.data = dh.data.float()

	# dh = one-h**2
	contractive_loss = torch.mm(dh**2,torch.sum(Variable(W)**2, dim=1).view(-1,1)).sum().mul_(lam)
	return mse + contractive_loss

def Validate(model,x,y):
	model.eval()
	recons_x = model(x)
	recons_x = recons_x.data.numpy()
	y = y.data.numpy()
	rmse = np.sqrt(np.square(y-recons_x).mean(axis=1)).mean()
	return rmse

def get_data_label(in_dir,win_size):
	train_file = os.listdir(in_dir)
	train_data = []
	train_label = []
	zero_arr = np.zeros((10,))
	padding = [zero_arr for i in range((win_size-1)/2)]
	half_win = (win_size-1)/2
	for file in train_file:
		arr = np.loadtxt(in_dir+"/"+file)[:,0:10]
		arr = np.vstack(padding+[arr]+padding)
		for i in range(half_win,len(arr)-half_win):
			train_data.append([arr[i-half_win:i],arr[i+1:i+1+half_win]])
			train_label.append(arr[i])
	train_data = np.array(train_data).reshape((-1,win_size-1,10))
	train_label = np.array(train_label).reshape((-1,10))
	return train_data,train_label

if __name__=="__main__":
	if sys.argv[1]=="train":
		win_size = 3
		print("reading data...")
		train_data,train_label = get_data_label("/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/src/dnn/lstm_data/train",win_size)
		test_data,test_label = get_data_label("/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/src/dnn/lstm_data/test",win_size)
		# print(test_data.shape)
		# print(test_label.shape)
		


		train_data = train_data[0:60000]
		train_label = train_label[0:60000]

		batch_size = 50
		batch_num = 60000/batch_size
		train_data = train_data.reshape((batch_num,batch_size,win_size-1,10))
		train_label = train_label.reshape((batch_num,batch_size,-1))

		train_data = torch.FloatTensor(train_data.tolist())
		train_label = torch.FloatTensor(train_label.tolist())

		test_data = torch.FloatTensor(test_data.tolist())
		test_label = torch.FloatTensor(test_label.tolist())

		test_data = Variable(test_data)
		test_label = Variable(test_label)


		mse_loss = nn.MSELoss()
		model = CAE(win_size)
		# model.load_state_dict(torch.load("best_model_con"))
		optimizer = optim.Adam(model.parameters(), lr = 0.001)
		# optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
		val_recons = None
		min_val_loss = 1000000
		beat_model = None
		print("begin training...")
		for epoch in range(1000):
			loss_val = 0
			for i in range(batch_num):
				train_data_batch = Variable(train_data[i])
				train_label_batch = Variable(train_label[i])

				optimizer.zero_grad()

				recons_x = model(train_data_batch)

				# loss = loss_function(W,train_batch,recons_x,h,mse_loss)z
				loss = mse_loss(recons_x, train_label_batch)
				loss.backward()
				optimizer.step()

				loss_val += loss.data[0]

			for param_group in optimizer.param_groups:
				param_group["lr"] *= 1

			print("Epoch "+str(epoch))
			print("train loss: "+str(loss_val/batch_num))
			val_loss = Validate(model,test_data,test_label)
			print("validation loss: "+str(val_loss))
			print("############################################")

			if min_val_loss > val_loss:
				min_val_loss = val_loss
				best_model = copy.deepcopy(model)
		torch.save(best_model.state_dict(),"best_model")

	elif sys.argv[1]=="encode":
		model = CAE()
		model.load_state_dict(torch.load("best_model_4"))
		weight1 = model.state_dict()['fc1.weight'].numpy()
		weight2 = model.state_dict()['fc2.weight'].numpy()
		aff = weight1.T.dot(weight2.T)
		show_heatmap(aff)
		exit()
		# new_arr = np.copy(weight)
		# new_arr[:,0] = weight[:,1]
		# new_arr[:,1] = weight[:,0]
		# print(np.around(new_arr,decimals=3))
		# show_heatmap(new_arr)

		test_data = np.loadtxt("../../seq_op/my_data/train_test_data/train_data/train_f0")
		test_data = torch.FloatTensor(test_data.tolist())
		test_data = Variable(test_data)

		enc_data = model.encoder(test_data)
		np.savetxt("train_encode_data",enc_data.data.numpy(),fmt="%.5f")

	elif sys.argv[1]=="decode":
		model = CAE()
		model.load_state_dict(torch.load("best_model_4"))

		encode_data = np.loadtxt("predict")
		encode_data = torch.FloatTensor(encode_data.tolist())
		encode_data = Variable(encode_data)

		dec_data = model.decoder(encode_data)
		np.savetxt("decode_predict",dec_data.data.numpy(),fmt="%.5f")

	elif sys.argv[1]=="visualize":
		model = CAE()
		model.load_state_dict(torch.load("best_model_hehe"))
		weight2 = model.state_dict()['fc2.weight'].numpy()
		weight4 = model.state_dict()['fc4.weight'].numpy()
		weight1 = model.state_dict()['fc1.weight'].numpy()
		weight3 = model.state_dict()['fc3.weight'].numpy()
		print(weight2)
		print(weight4)
		print(weight1)
		print(weight3)
		# new_arr = np.copy(weight)
		# new_arr[:,0] = weight[:,1]
		# new_arr[:,1] = weight[:,0]
		# print(np.around(new_arr,decimals=3))
		# show_heatmap(new_arr)

		encode_data = np.loadtxt("train_encode_data")
		# sample_idx = 910
		# sample = encode_data[sample_idx,:]

		# for i in range(len(sample)):
		# 	plt.plot(weight[:,i]*sample[i],label="dimension "+str(i))
		# plt.legend()
		# plt.savefig(str(sample_idx)+".jpg")

		# plt.plot(encode_data[0][0]*weight[:,0],label=str(0)+": pei2")
		# plt.plot(encode_data[1][0]*weight[:,0],label=str(1)+": tong2")
		# plt.plot(encode_data[839][0]*weight[:,0],label=str(839)+": tong1")
		# plt.plot(encode_data[910][0]*weight[:,0],label=str(910)+": tong4")
		# plt.plot(encode_data[1000][0]*weight[:,0],label=str(1000)+": guo2")
		# plt.plot(encode_data[200][0]*weight[:,0],label=str(200)+": tuan2")
		# plt.legend()
		# plt.show()
		# next, use decision tree to deal with relationship between the individual values and the coefficients





















