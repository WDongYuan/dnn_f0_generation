cuda_flag = None
##concat? element-wise product? add?
##try the last hidden layer
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
from scipy.fftpack import idct, dct
###########################################################
#GPU OPTION
###########################################################
# if cuda_flag:
# 	import torch.backends.cudnn as cudnn
###########################################################

class DCT_LSTM(nn.Module):
	def __init__(self,emb_size,pos_emb_size,feat_size,voc_size,pos_num,lstm_hidden_size,dct_num,linear_h1):
		super(DCT_LSTM, self).__init__()
		self.emb_size = emb_size
		self.feat_size = feat_size
		self.pos_emb_size = pos_emb_size
		self.lstm_hidden_size = lstm_hidden_size
		self.dct_num = dct_num
		self.linear_h1 = linear_h1
		self.voc_size = voc_size
		self.pos_num = pos_num
		self.batch_size = -1
		self.max_length = -1

		self.embed = nn.Embedding(self.voc_size, self.emb_size,padding_idx=0)
		init.uniform(self.embed.weight,a=-0.01,b=0.01)
		self.pos_embed = nn.Embedding(self.pos_num, self.pos_emb_size,padding_idx=0)
		init.uniform(self.embed.weight,a=-0.01,b=0.01)

		##LSTM
		self.lstm_layer = 1
		self.bidirectional_flag = True
		self.direction = 2 if self.bidirectional_flag else 1
		self.emb_lstm = nn.LSTM(self.emb_size+self.pos_emb_size+self.feat_size, self.lstm_hidden_size,
			num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)


		self.non_linear = nn.ReLU()
		self.emb_l1 = nn.Linear(self.lstm_hidden_size*self.direction,self.linear_h1)
		self.linear_init(self.emb_l1)
		self.emb_l2 = nn.Linear(self.linear_h1,self.dct_num)
		self.linear_init(self.emb_l2)
		self.tanh = nn.Tanh()


	def linear_init(self,layer,lower=-1,upper=1):
		layer.weight.data.uniform_(lower, upper)
		layer.bias.data.uniform_(lower, upper)
	def init_hidden(self):
		direction = 2 if self.bidirectional_flag else 1
		###########################################################
		#GPU OPTION
		###########################################################
		if cuda_flag:
			return Variable(torch.rand(self.lstm_layer*direction,self.batch_size,self.lstm_hidden_size).cuda(async=True))
		else:
			return Variable(torch.rand(self.lstm_layer*direction,self.batch_size,self.lstm_hidden_size))
		###########################################################

	def forward(self,sents,pos,feat,sent_length):
		self.batch_size,self.max_length = sents.size()
		emb = self.embed(sents)
		pos = self.pos_embed(pos)

		emb = torch.cat((emb,pos,feat),dim=2)

		c_0 = self.init_hidden()
		h_0 = self.init_hidden()
		emb_h_n, (emb_h_t,emb_c_t) = self.emb_lstm(emb,(h_0,c_0))
		emb_h_t = emb_h_t.permute(1,0,2).contiguous().view(self.batch_size,self.direction*self.lstm_hidden_size)

		emb_h = self.emb_l1(emb_h_t)
		emb_h = self.tanh(emb_h)
		emb_h = self.emb_l2(emb_h)

		h = emb_h

		h = h.view(self.batch_size,self.dct_num)
		return h

def Train(train_emb,train_pos,train_feat,train_f0,train_len,val_emb,val_pos,val_feat,val_f0,val_len,\
	model,optimizer,learning_rate,decay_step,decay_rate,epoch_num):
	###########################################################
	#GPU OPTION
	###########################################################
	if cuda_flag:
		torch.backends.cudnn.benchmark = True
		model.cuda()
	###########################################################
	LF = nn.MSELoss()
	min_loss = 100000000
	print("begin training...")
	for epoch in range(epoch_num):
		start_time = time.time()
		loss_val = 0
		for i in range(len(train_emb)):
			###########################################################
			#GPU OPTION
			###########################################################
			if cuda_flag:
				train_emb_batch = Variable(train_emb[i].cuda(async=True))
				train_pos_batch = Variable(train_pos[i].cuda(async=True))
				train_feat_batch = Variable(train_feat[i].cuda(async=True))
				train_f0_batch = Variable(train_f0[i].cuda(async=True))
				train_len_batch = Variable(train_len[i].cuda(async=True))
			else:
				train_emb_batch = Variable(train_emb[i])
				train_pos_batch = Variable(train_pos[i])
				train_feat_batch = Variable(train_feat[i])
				train_f0_batch = Variable(train_f0[i])
				train_len_batch = Variable(train_len[i])
			###########################################################


			optimizer.zero_grad()
			outputs = model(train_emb_batch,train_pos_batch,train_feat_batch,train_len_batch)
			# print(outputs.size())
			# print(train_label_batch.size())
			loss = LF(outputs,train_f0_batch)
			loss.backward()
			optimizer.step()
			loss_val += loss.data[0]
		if (epoch+1)%1==0:
			print("Epoch "+str(epoch))
			print("train loss: "+str(loss_val/len(train_emb)))
			val_loss,result = Validate(model,val_emb,val_pos,val_feat,val_f0,val_len)
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

def Validate(model,val_emb,val_pos,val_feat,val_f0,val_len,save_prediction=""):
	model.eval()
	val_f0_shape = val_f0.size()
	batch_size = val_f0_shape[0]

	###########################################################
	#GPU OPTION
	###########################################################
	if cuda_flag:
		result = model(Variable(val_emb.cuda(async=True)),Variable(val_pos.cuda(async=True)),Variable(val_feat.cuda(async=True)),
			Variable(val_len.cuda(async=True))).data.cpu().numpy().reshape((batch_size,model.dct_num))
		val_f0 = val_f0.cpu().numpy().reshape((batch_size,model.dct_num))
	else:
		result = model(Variable(val_emb),Variable(val_pos),Variable(val_feat),Variable(val_len)).data.numpy().reshape((batch_size,model.dct_num))
		val_f0 = val_f0.numpy().reshape((batch_size,model.dct_num))
	###########################################################
	val_len = val_len.numpy()

	loss = np.sqrt(np.square(val_f0-result).mean(axis=1)).mean()
	if save_prediction!="":
		mean_std = np.loadtxt("./tmp_mean_std",delimiter=" ")
		row = 0
		prediction = np.zeros((val_len.sum(),10))
		for i in range(len(val_len)):
			tmp_arr = np.zeros((val_len[i]*10,))
			# tmp_arr[0:model.dct_num] = result[i]
			tmp_arr[0:model.dct_num] = result[i]*mean_std[1]+mean_std[0]
			tmp_arr = idct(tmp_arr)/(2*len(tmp_arr))
			tmp_arr = tmp_arr.reshape((val_len[i],10))
			prediction[row:row+val_len[i]] = tmp_arr
			row += val_len[i]
		np.savetxt(save_prediction,prediction,delimiter=" ",fmt="%.3f")

	return loss,result.reshape((val_f0_shape))




