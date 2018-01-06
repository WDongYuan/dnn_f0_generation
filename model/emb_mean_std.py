import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.nn.init import kaiming_uniform
import torch.nn.init as init
import time
class EMB_MEAN_STD(nn.Module):
	def __init__(self,emb_size,voc_size,lstm_hidden_size,f0_dim,linear_shape,linear_mean,linear_std):
		super(EMB_MEAN_STD, self).__init__()
		self.emb_size = emb_size
		self.lstm_hidden_size = lstm_hidden_size
		self.f0_dim = f0_dim

		self.linear_shape = linear_shape
		self.linear_mean = linear_mean
		self.linear_std = linear_std

		self.voc_size = voc_size
		self.batch_size = -1
		self.max_length = -1

		self.embed = nn.Embedding(self.voc_size, self.emb_size,padding_idx=0)
		init.uniform(self.embed.weight,a=-0.01,b=0.01)



		##LSTM
		self.lstm_layer = 1
		self.bidirectional_flag = True
		self.direction = 2 if self.bidirectional_flag else 1

		self.shape_lstm = nn.LSTM(self.emb_size, self.lstm_hidden_size,
			num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
		self.mean_lstm = nn.LSTM(self.emb_size, self.lstm_hidden_size,
			num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
		self.std_lstm = nn.LSTM(self.emb_size, self.lstm_hidden_size,
			num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)


		# self.non_linear = nn.Tanh()
		# self.non_linear = nn.Sigmoid()
		self.non_linear = nn.ReLU()
		self.shape_l1 = nn.Linear(self.lstm_hidden_size*self.direction,self.linear_shape)
		self.linear_init(self.shape_l1)
		self.shape_l2 = nn.Linear(self.linear_shape,self.f0_dim)
		self.linear_init(self.shape_l2)

		self.mean_l1 = nn.Linear(self.lstm_hidden_size*self.direction,self.linear_mean)
		self.linear_init(self.mean_l1)
		self.mean_l2 = nn.Linear(self.linear_mean,1)
		self.linear_init(self.mean_l2)

		self.std_l1 = nn.Linear(self.lstm_hidden_size*self.direction,self.linear_std)
		self.linear_init(self.std_l1)
		self.std_l2 = nn.Linear(self.linear_std,1)
		self.linear_init(self.std_l2)


	def linear_init(self,layer):
		layer.weight.data.uniform_(-1, 1)
		layer.bias.data.uniform_(-1, 1)
	def init_hidden(self):
		direction = 2 if self.bidirectional_flag else 1
		return Variable(torch.rand(self.lstm_layer*direction,self.batch_size,self.lstm_hidden_size))

	def forward(self,sents,sent_length):
		self.batch_size,self.max_length = sents.size()
		emb = self.embed(sents)
		# emb = self.embed_linear(emb)

		c_0 = self.init_hidden()
		h_0 = self.init_hidden()

		h_n_shape, (h_t,c_t) = self.shape_lstm(emb,(h_0,c_0))
		h_n_mean, (h_t,c_t) = self.mean_lstm(emb,(h_0,c_0))
		h_n_std, (h_t,c_t) = self.std_lstm(emb,(h_0,c_0))

		h_shape = self.shape_l1(h_n_shape)
		h_shape = self.non_linear(h_shape)
		h_shape = self.shape_l2(h_shape)

		h_mean = self.mean_l1(h_n_mean)
		h_mean = self.non_linear(h_mean)
		h_mean = self.mean_l2(h_mean)

		h_std = self.std_l1(h_n_std)
		h_std = self.non_linear(h_std)
		h_std = self.std_l2(h_std)

		h = torch.mul(h_shape,h_std)+h_mean



		h = h.view(self.batch_size,self.max_length*self.f0_dim)
		return h

def Train(train_emb,train_f0,train_len,val_emb,val_f0,val_len,\
	model,optimizer,learning_rate,decay_step,decay_rate,epoch_num):
	# print(train_emb)
	LF = nn.MSELoss()
	min_loss = 100000000
	print("begin training...")
	for epoch in range(epoch_num):
		start_time = time.time()
		loss_val = 0
		for i in range(len(train_emb)):
			# if i==6:
			# 	return
			train_emb_batch = Variable(train_emb[i])
			train_f0_batch = Variable(train_f0[i])
			train_len_batch = Variable(train_len[i])

			optimizer.zero_grad()
			outputs = model(train_emb_batch,train_len_batch)
			# print(outputs.size())
			# print(train_label_batch.size())
			loss = LF(outputs,train_f0_batch)
			loss.backward()
			optimizer.step()
			loss_val += loss.data[0]
		if (epoch+1)%1==0:
			print("Epoch "+str(epoch))
			print("train loss: "+str(loss_val/len(train_emb)))
			val_loss,result = Validate(model,val_emb,val_f0,val_len)
			print("val loss: "+str(val_loss))
			if val_loss<min_loss:
				torch.save(model,"./my_best_model_.model")
				min_loss = val_loss
		if (epoch+1)%decay_step==0:
			learning_rate *= decay_rate
			for param_group in optimizer.param_groups:
				param_group['lr'] = learning_rate
			print("#####################################")
			print("learning rate: "+str(learning_rate))
			print("#####################################")
		print("time: "+str(time.time()-start_time))
		print("#####################################")

def Validate(model,val_emb,val_f0,val_len,save_prediction=""):
	model.eval()
	val_f0_shape = val_f0.size()
	batch_size = val_f0_shape[0]

	result = model(Variable(val_emb),Variable(val_len)).data.numpy().reshape((batch_size,model.max_length,model.f0_dim))
	val_f0 = val_f0.numpy().reshape((batch_size,model.max_length,model.f0_dim))
	val_len = val_len.numpy()
	loss = []

	prediction = np.zeros((np.sum(val_len),model.f0_dim))
	true_f0 = np.zeros((np.sum(val_len),model.f0_dim))
	row_count = 0
	for i in range(batch_size):
		tmp_result = result[i,0:val_len[i],:]
		tmp_f0 = val_f0[i,0:val_len[i],:]
		prediction[row_count:row_count+val_len[i]] = tmp_result
		true_f0[row_count:row_count+val_len[i]] = tmp_f0
		row_count += val_len[i]

	loss = np.sqrt(np.square(prediction-true_f0).mean(axis=1)).mean()

	if save_prediction!="":
		np.savetxt(save_prediction,prediction,delimiter=" ",fmt="%.3f")

	return loss,result.reshape((val_f0_shape))




