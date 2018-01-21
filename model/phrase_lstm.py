# -*- coding: utf-8 -*-
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
###########################################################
#GPU OPTION
###########################################################
if cuda_flag:
	import torch.backends.cudnn as cudnn
###########################################################

class PHRASE_LSTM(nn.Module):
	def __init__(self,emb_size,pos_emb_size,tone_emb_size,
		cons_num,vowel_num,pretone_num,tone_num,postone_num,feat_size,phrase_num,voc_size,pos_num,
		lstm_hidden_size,f0_dim,linear_h1):
		super(PHRASE_LSTM, self).__init__()
		self.emb_size = emb_size
		self.feat_size = feat_size
		self.pos_emb_size = pos_emb_size
		self.tone_emb_size = tone_emb_size
		self.phrase_num = phrase_num

		self.pretone_num = pretone_num
		self.tone_num = tone_num
		self.postone_num = postone_num
		self.cons_num = cons_num
		self.vowel_num = vowel_num

		self.lstm_hidden_size = lstm_hidden_size
		self.f0_dim = f0_dim
		self.linear_h1 = linear_h1
		self.voc_size = voc_size
		self.pos_num = pos_num
		self.batch_size = -1
		self.max_length = -1

		self.phrase_hidden_size = self.lstm_hidden_size
		self.phrase_linear_size = self.linear_h1

		self.embed = nn.Embedding(self.voc_size, self.emb_size,padding_idx=0)
		init.uniform(self.embed.weight,a=-0.01,b=0.01)
		self.pos_embed = nn.Embedding(self.pos_num, self.pos_emb_size,padding_idx=0)
		init.uniform(self.pos_embed.weight,a=-0.01,b=0.01)

		self.tone_embed = nn.Embedding(self.tone_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.tone_embed.weight,a=-0.01,b=0.01)
		self.pretone_embed = nn.Embedding(self.pretone_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.pretone_embed.weight,a=-0.01,b=0.01)
		self.postone_embed = nn.Embedding(self.postone_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.postone_embed.weight,a=-0.01,b=0.01)

		self.cons_embed = nn.Embedding(self.cons_num, self.emb_size,padding_idx=0)
		init.uniform(self.cons_embed.weight,a=-0.01,b=0.01)
		self.vowel_embed = nn.Embedding(self.vowel_num, self.emb_size,padding_idx=0)
		init.uniform(self.vowel_embed.weight,a=-0.01,b=0.01)

		##LSTM
		self.lstm_layer = 2
		self.bidirectional_flag = True
		self.direction = 2 if self.bidirectional_flag else 1
		# self.emb_lstm = nn.LSTM(self.emb_size+self.pos_emb_size, self.lstm_hidden_size,
		# 	num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
		self.feat_lstm = nn.LSTM(self.feat_size+self.emb_size+self.pos_emb_size, self.lstm_hidden_size,
			num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
		self.phrase_lstm = nn.LSTM(self.phrase_num+self.tone_emb_size+2*emb_size, self.phrase_hidden_size,
			num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
		# self.syl_lstm = nn.LSTM(3*self.tone_emb_size, self.lstm_hidden_size,
		# 	num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)


		self.non_linear = nn.ReLU()
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()

		# self.emb_l1 = nn.Linear(self.lstm_hidden_size*self.direction,self.linear_h1)
		# self.linear_init(self.emb_l1)
		# self.emb_l2 = nn.Linear(self.linear_h1,self.f0_dim)
		# self.linear_init(self.emb_l2)

		self.feat_l1 = nn.Linear(self.lstm_hidden_size*self.direction,self.linear_h1)
		self.linear_init(self.feat_l1)
		self.feat_l2 = nn.Linear(self.linear_h1,self.f0_dim)
		self.linear_init(self.feat_l2)

		self.phrase_l1 = nn.Linear(self.phrase_hidden_size*self.direction,self.phrase_linear_size)
		self.linear_init(self.phrase_l1)
		self.phrase_l2 = nn.Linear(self.phrase_linear_size,self.f0_dim)
		self.linear_init(self.phrase_l2)

		# self.syl_l1 = nn.Linear(self.lstm_hidden_size*self.direction,self.linear_h1)
		# self.linear_init(self.syl_l1)
		# self.syl_l2 = nn.Linear(self.linear_h1,self.f0_dim)
		# self.linear_init(self.syl_l2)

		# self.mean_l1 = nn.Linear(self.lstm_hidden_size*self.direction,self.linear_h1)


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

	def init_phrase_hidden(self):
		direction = 2 if self.bidirectional_flag else 1
		###########################################################
		#GPU OPTION
		###########################################################
		if cuda_flag:
			return Variable(torch.rand(self.lstm_layer*direction,self.batch_size,self.phrase_hidden_size).cuda(async=True))
		else:
			return Variable(torch.rand(self.lstm_layer*direction,self.batch_size,self.phrase_hidden_size))
		###########################################################

	def forward(self,sents,pos,cons,vowel,pretone,tone,postone,feat,phrase,sent_length):
		self.batch_size,self.max_length = sents.size()
		emb = self.embed(sents)
		pos = self.pos_embed(pos)
		# pretone = self.pretone_embed(pretone)
		tone = self.tone_embed(tone)
		# postone = self.postone_embed(postone)
		cons = self.cons_embed(cons)
		vowel = self.vowel_embed(vowel)


		c_0 = self.init_hidden()
		h_0 = self.init_hidden()

		feat = torch.cat((emb,pos,feat),dim=2)
		feat_h_n, (_,_) = self.feat_lstm(feat,(h_0,c_0))
		feat_h = self.feat_l1(feat_h_n)
		feat_h = self.tanh(feat_h)
		feat_h = self.feat_l2(feat_h)

		c_0 = self.init_phrase_hidden()
		h_0 = self.init_phrase_hidden()

		ph = torch.cat((phrase,tone,cons,vowel),dim=2)
		ph_h_n, (_,_) = self.phrase_lstm(ph,(h_0,c_0))
		ph_h = self.phrase_l1(ph_h_n)
		ph_h = self.relu(ph_h)
		ph_h = self.phrase_l2(ph_h)

		h = feat_h+ph_h
		# h = feat_h

		h = h.view(self.batch_size,self.max_length*self.f0_dim)
		################################################################################
		# feat_h = feat_h.view(self.batch_size,self.max_length*self.f0_dim)
		# ph_h = ph_h.view(self.batch_size,self.max_length*self.f0_dim)
		# return h,feat_h,ph_h
		################################################################################
		return h

class PHRASE_MEAN_LSTM(nn.Module):
	def __init__(self,emb_size,pos_emb_size,tone_emb_size,
		cons_num,vowel_num,pretone_num,tone_num,postone_num,feat_size,phrase_num,voc_size,pos_num,
		lstm_hidden_size,f0_dim,linear_h1):
		super(PHRASE_MEAN_LSTM, self).__init__()
		self.emb_size = emb_size
		self.feat_size = feat_size
		self.pos_emb_size = pos_emb_size
		self.tone_emb_size = tone_emb_size
		self.phrase_num = phrase_num

		self.pretone_num = pretone_num
		self.tone_num = tone_num
		self.postone_num = postone_num
		self.cons_num = cons_num
		self.vowel_num = vowel_num

		self.lstm_hidden_size = lstm_hidden_size
		self.f0_dim = 1
		self.linear_h1 = linear_h1
		self.voc_size = voc_size
		self.pos_num = pos_num
		self.batch_size = -1
		self.max_length = -1

		self.phrase_hidden_size = self.lstm_hidden_size
		self.phrase_linear_size = self.linear_h1

		self.embed = nn.Embedding(self.voc_size, self.emb_size,padding_idx=0)
		init.uniform(self.embed.weight,a=-0.01,b=0.01)
		self.pos_embed = nn.Embedding(self.pos_num, self.pos_emb_size,padding_idx=0)
		init.uniform(self.pos_embed.weight,a=-0.01,b=0.01)

		self.tone_embed = nn.Embedding(self.tone_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.tone_embed.weight,a=-0.01,b=0.01)
		self.pretone_embed = nn.Embedding(self.pretone_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.pretone_embed.weight,a=-0.01,b=0.01)
		self.postone_embed = nn.Embedding(self.postone_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.postone_embed.weight,a=-0.01,b=0.01)

		self.cons_embed = nn.Embedding(self.cons_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.cons_embed.weight,a=-0.01,b=0.01)
		self.vowel_embed = nn.Embedding(self.vowel_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.vowel_embed.weight,a=-0.01,b=0.01)

		##LSTM
		self.lstm_layer = 1
		self.bidirectional_flag = True
		self.direction = 2 if self.bidirectional_flag else 1
		self.emb_lstm = nn.LSTM(self.emb_size+self.pos_emb_size, self.lstm_hidden_size,
			num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
		self.feat_lstm = nn.LSTM(self.feat_size, self.lstm_hidden_size,
			num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)


		self.non_linear = nn.ReLU()
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()

		self.emb_l1 = nn.Linear(self.lstm_hidden_size*self.direction,self.linear_h1)
		self.linear_init(self.emb_l1)
		self.emb_l2 = nn.Linear(self.linear_h1,self.f0_dim)
		self.linear_init(self.emb_l2)

		self.feat_l1 = nn.Linear(self.lstm_hidden_size*self.direction,self.linear_h1)
		self.linear_init(self.feat_l1)
		self.feat_l2 = nn.Linear(self.linear_h1,self.f0_dim)
		self.linear_init(self.feat_l2)


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

	def init_phrase_hidden(self):
		direction = 2 if self.bidirectional_flag else 1
		###########################################################
		#GPU OPTION
		###########################################################
		if cuda_flag:
			return Variable(torch.rand(self.lstm_layer*direction,self.batch_size,self.phrase_hidden_size).cuda(async=True))
		else:
			return Variable(torch.rand(self.lstm_layer*direction,self.batch_size,self.phrase_hidden_size))
		###########################################################

	def forward(self,sents,pos,cons,vowel,pretone,tone,postone,feat,phrase,sent_length):
		self.batch_size,self.max_length = sents.size()
		emb = self.embed(sents)
		pos = self.pos_embed(pos)
		tone = self.tone_embed(tone)
		cons = self.cons_embed(cons)
		vowel = self.vowel_embed(vowel)


		c_0 = self.init_hidden()
		h_0 = self.init_hidden()

		emb = torch.cat((emb,pos),dim=2)
		emb_h_n, (_,_) = self.emb_lstm(emb,(h_0,c_0))
		emb_h = self.emb_l1(emb_h_n)
		emb_h = self.tanh(emb_h)
		emb_h = self.emb_l2(emb_h)

		# feat_h_n, (_,_) = self.feat_lstm(feat,(h_0,c_0))
		# feat_h = self.feat_l1(feat_h_n)
		# feat_h = self.relu(feat_h)
		# feat_h = self.feat_l2(feat_h)

		h = emb_h

		h = h.view(self.batch_size,self.max_length*self.f0_dim)
		return h




def Train(train_emb,train_pos,train_cons,train_vowel,train_pretone,train_tone,train_postone,train_feat,train_phrase,train_f0,train_len,
	val_emb,val_pos,val_cons,val_vowel,val_pretone,val_tone,val_postone,val_feat,val_phrase,val_f0,val_len,
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
				train_cons_batch = Variable(train_cons[i].cuda(async=True))
				train_vowel_batch = Variable(train_vowel[i].cuda(async=True))
				train_tone_batch = Variable(train_tone[i].cuda(async=True))
				train_pretone_batch = Variable(train_pretone[i].cuda(async=True))
				train_postone_batch = Variable(train_postone[i].cuda(async=True))
				train_feat_batch = Variable(train_feat[i].cuda(async=True))
				train_f0_batch = Variable(train_f0[i].cuda(async=True))
				train_len_batch = Variable(train_len[i].cuda(async=True))
				train_phrase_batch = Variable(train_phrase[i].cuda(async=True))
			else:
				train_emb_batch = Variable(train_emb[i])
				train_pos_batch = Variable(train_pos[i])
				train_cons_batch = Variable(train_cons[i])
				train_vowel_batch = Variable(train_vowel[i])
				train_tone_batch = Variable(train_tone[i])
				train_pretone_batch = Variable(train_pretone[i])
				train_postone_batch = Variable(train_postone[i])
				train_feat_batch = Variable(train_feat[i])
				train_f0_batch = Variable(train_f0[i])
				train_len_batch = Variable(train_len[i])
				train_phrase_batch = Variable(train_phrase[i])
			###########################################################


			optimizer.zero_grad()
			outputs = model(train_emb_batch,train_pos_batch,train_cons_batch,train_vowel_batch,
				train_pretone_batch,train_tone_batch,train_postone_batch,train_feat_batch,train_phrase_batch,train_len_batch)
			# print(outputs.size())
			# print(train_label_batch.size())
			loss = LF(outputs,train_f0_batch)
			loss.backward()
			optimizer.step()
			loss_val += loss.data[0]
		if (epoch+1)%1==0:
			print("Epoch "+str(epoch))
			print("train loss: "+str(loss_val/len(train_emb)))
			val_loss,result = Validate(model,val_emb,val_pos,val_cons,val_vowel,val_pretone,val_tone,val_postone,val_feat,val_phrase,val_f0,val_len)
			print("val loss: "+str(val_loss))
			if val_loss<min_loss:
				torch.save(model,"./my_best_model.model")
				min_loss = val_loss
		if (epoch+1)%decay_step==0:
			learning_rate *= decay_rate
			# print(len(optimizer.param_groups))
			for param_group in optimizer.param_groups:
				param_group['lr'] = learning_rate
			print("#####################################")
			print("learning rate: "+str(learning_rate))
			print("#####################################")
		print("time: "+str(time.time()-start_time))
		print("#####################################")

def Validate(model,val_emb,val_pos,val_cons,val_vowel,val_pretone,val_tone,val_postone,val_feat,val_phrase,val_f0,val_len,save_prediction=""):
	model.eval()
	val_f0_shape = val_f0.size()
	batch_size = val_f0_shape[0]

	###########################################################
	#GPU OPTION
	###########################################################
	if cuda_flag:
		result = model(Variable(val_emb.cuda(async=True)),Variable(val_pos.cuda(async=True)),Variable(val_cons.cuda(async=True)),Variable(val_vowel.cuda(async=True)),
			Variable(val_pretone.cuda(async=True)),Variable(val_tone.cuda(async=True)),Variable(val_postone.cuda(async=True)),
			Variable(val_feat.cuda(async=True)),Variable(val_phrase.cuda(async=True)),Variable(val_len.cuda(async=True))).data.cpu().numpy().reshape((batch_size,model.max_length,model.f0_dim))
		val_f0 = val_f0.cpu().numpy().reshape((batch_size,model.max_length,model.f0_dim))
	else:
		# result,res_h,emb_h = model(Variable(val_emb),Variable(val_pos),Variable(val_cons),Variable(val_vowel),
		# 	Variable(val_pretone),Variable(val_tone),Variable(val_postone),
		# 	Variable(val_feat),Variable(val_phrase),Variable(val_len))
		result = model(Variable(val_emb),Variable(val_pos),Variable(val_cons),Variable(val_vowel),
			Variable(val_pretone),Variable(val_tone),Variable(val_postone),
			Variable(val_feat),Variable(val_phrase),Variable(val_len))
		result = result.data.numpy().reshape((batch_size,model.max_length,model.f0_dim))
		val_f0 = val_f0.numpy().reshape((batch_size,model.max_length,model.f0_dim))
		################################################################################
		# val_len = val_len.numpy()
		# emb_h = emb_h.data.numpy().reshape((batch_size,model.max_length,model.f0_dim))
		# res_h = res_h.data.numpy().reshape((batch_size,model.max_length,model.f0_dim))
		# tmp_emb_h = np.zeros((np.sum(val_len),model.f0_dim))
		# tmp_res_h = np.zeros((np.sum(val_len),model.f0_dim))
		# row_count = 0
		# for i in range(batch_size):
		# 	tmp_emb = emb_h[i,0:val_len[i],:]
		# 	tmp_res = res_h[i,0:val_len[i],:]
		# 	tmp_emb_h[row_count:row_count+val_len[i]] = tmp_emb
		# 	tmp_res_h[row_count:row_count+val_len[i]] = tmp_res
		# 	row_count += val_len[i]
		# np.savetxt("emb_h",tmp_emb_h,delimiter=" ")
		# np.savetxt("res_h",tmp_res_h,delimiter=" ")
		# exit()
		################################################################################
	###########################################################
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




