# coding=utf-8
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
		cons_num,vowel_num,pretone_num,tone_num,postone_num,feat_size,phrase_num,dep_num,voc_size,pos_num,pos_feat_num,
		lstm_hidden_size,f0_dim,linear_h1):
		super(PHRASE_LSTM, self).__init__()
		self.emb_size = emb_size
		self.feat_size = feat_size
		self.pos_emb_size = pos_emb_size
		self.pos_emb_length = 3##how many pos emb per sample (pre,current,post)
		self.tone_emb_size = tone_emb_size
		self.phrase_num = phrase_num
		self.dep_num = dep_num
		self.dep_lemb_size = 20
		self.emb_l_size = 10
		self.grad_emb_size = 10

		self.pretone_num = pretone_num
		self.tone_num = tone_num
		self.postone_num = postone_num
		self.cons_num = cons_num
		self.vowel_num = vowel_num

		self.lstm_hidden_size = lstm_hidden_size
		self.f0_dim = f0_dim
		# self.f0_dim = 1
		self.linear_h1 = linear_h1
		self.voc_size = voc_size
		self.pos_num = pos_num
		self.pos_feat_num = pos_feat_num
		self.batch_size = -1
		self.max_length = -1

		self.phrase_hidden_size = self.lstm_hidden_size
		self.phrase_linear_size = self.linear_h1

		self.grad_embed = nn.Embedding(self.voc_size, self.grad_emb_size,padding_idx=0)
		init.uniform(self.grad_embed.weight,a=-0.01,b=0.01)

		self.embed = self.get_embedding("./lstm_data/pretrain_emb",self.voc_size,self.emb_size)

		self.pos_embed = nn.Embedding(self.pos_num, self.pos_emb_size,padding_idx=0)
		init.uniform(self.pos_embed.weight,a=-0.01,b=0.01)

		self.tone_embed = nn.Embedding(self.tone_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.tone_embed.weight,a=-0.01,b=0.01)

		self.cons_embed = nn.Embedding(self.cons_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.cons_embed.weight,a=-0.01,b=0.01)
		self.vowel_embed = nn.Embedding(self.vowel_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.vowel_embed.weight,a=-0.01,b=0.01)
		# self.vowel_ch_embed = nn.Embedding(self.vowel_ch_num, self.tone_emb_size,padding_idx=0)
		# init.uniform(self.vowel_ch_embed.weight,a=-0.01,b=0.01)

		##LSTM
		self.lstm_layer = 1
		self.bidirectional_flag = True
		self.direction = 2 if self.bidirectional_flag else 1
		# self.emb_lstm = nn.LSTM(self.emb_size+self.pos_emb_size, self.lstm_hidden_size,
		# 	num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
		self.feat_lstm = nn.LSTM(self.emb_l_size+self.feat_size+self.pos_emb_length*self.pos_emb_size+self.pos_feat_num,self.lstm_hidden_size,
			num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
		# self.feat_lstm = nn.LSTM(self.grad_emb_size,self.lstm_hidden_size,
		# 	num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)

		self.phrase_lstm_layer = 1
		self.phrase_bidirectional_flag = True
		self.phrase_direction = 2 if self.phrase_bidirectional_flag else 1
		self.phrase_lstm = nn.LSTM(3*self.tone_emb_size+self.feat_size+self.phrase_num, self.phrase_hidden_size,
			num_layers=self.phrase_lstm_layer,bidirectional=self.phrase_bidirectional_flag,batch_first=True)


		self.non_linear = nn.ReLU()
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()

		self.emb_l1 = nn.Linear(self.emb_size,self.emb_l_size)
		# self.linear_init(self.emb_l1)
		# self.emb_l2 = nn.Linear(self.linear_h1,self.f0_dim)
		# self.linear_init(self.emb_l2)
		self.dep_lemb = nn.Linear(self.dep_num,self.dep_lemb_size)
		self.linear_init(self.dep_lemb)

		self.feat_l1 = nn.Linear(self.lstm_hidden_size*self.direction,self.linear_h1)
		self.linear_init(self.feat_l1)
		self.feat_l2 = nn.Linear(self.linear_h1,self.f0_dim)
		self.linear_init(self.feat_l2)

		self.phrase_l1 = nn.Linear(self.phrase_hidden_size*self.phrase_direction,self.phrase_linear_size)
		self.linear_init(self.phrase_l1)
		self.phrase_l2 = nn.Linear(self.phrase_linear_size,self.f0_dim)
		self.linear_init(self.phrase_l2)

		self.comb_l1 = nn.Linear(2*self.f0_dim,2*self.f0_dim)
		self.comb_l2 = nn.Linear(2*self.f0_dim,self.f0_dim)


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
		direction = 2 if self.phrase_bidirectional_flag else 1
		###########################################################
		#GPU OPTION
		###########################################################
		if cuda_flag:
			return Variable(torch.rand(self.phrase_lstm_layer*direction,self.batch_size,self.phrase_hidden_size).cuda(async=True))
		else:
			return Variable(torch.rand(self.phrase_lstm_layer*direction,self.batch_size,self.phrase_hidden_size))
		###########################################################

	def get_embedding(self,emb_file,voc_size,emb_size):
		arr = np.loadtxt(emb_file)
		embed = nn.Embedding(voc_size, emb_size)
		embed.weight.data.copy_(torch.from_numpy(arr))
		embed.weight.requires_grad = False
		return embed

	def get_self_f0_delta(self,data):
		batch_size,max_length,f0_dim = data.size()
		delta = data[:,:,1:f0_dim]-data[:,:,0:f0_dim-1]
		# delta = Variable(delta)
		delta_length = f0_dim-1
		return delta,delta_length
	def get_mean_delta(self,data):
		batch_size,max_length,f0_dim = data.size()
		mean = torch.mean(data,dim=2)
		if cuda_flag:
			delta = Variable(torch.zeros(batch_size,max_length,3).cuda(async=True))
		else:
			delta = Variable(torch.zeros(batch_size,max_length,3))
		delta[:,1:max_length,0] = mean[:,1:max_length]-mean[:,0:max_length-1]
		delta[:,0:max_length-1,1] = mean[:,1:max_length]-mean[:,0:max_length-1]
		delta[:,:,2] = delta[:,:,1]-delta[:,:,0]
		delta_length = 3
		# delta = Variable(delta)
		# print(delta.size())
		return delta,delta_length

	def get_f0_delta(self,data):
		batch_size,max_length,f0_dim = data.size()
		if cuda_flag:
			delta = Variable(torch.zeros(batch_size,max_length,3,f0_dim).cuda(async=True))
		else:
			delta = Variable(torch.zeros(batch_size,max_length,2,f0_dim))
		delta[:,1:max_length,0,:] = data[:,1:max_length,:]-data[:,0:max_length-1,:]
		delta[:,0:max_length-1,1,:] = data[:,1:max_length,:]-data[:,0:max_length-1,:]
		delta[:,:,2,:] = delta[:,:,1,:]-delta[:,:,0,:]
		delta = delta.view(batch_size,max_length,3*f0_dim)
		delta_length = 3*f0_dim
		# delta = Variable(delta)
		# print(delta.size())
		return delta,delta_length

	def forward(self,sents,pos,pos_feat,cons,vowel,pretone,tone,postone,feat,phrase,dep,sent_length):
		self.batch_size,self.max_length = sents.size()
		emb = self.embed(sents)
		grad_emb = self.grad_embed(sents)

		pos = self.pos_embed(pos.view(self.batch_size,self.max_length*self.pos_emb_length))
		pos = pos.view(self.batch_size,self.max_length,self.pos_emb_length*self.pos_emb_size)

		tone = self.tone_embed(tone)
		cons = self.cons_embed(cons)

		# vowel_ch = vowel[:,:,1:].contiguous()
		# vowel_ch = self.vowel_ch_embed(vowel_ch.view(self.batch_size,self.max_length*4))
		# vowel_ch = vowel_ch.view(self.batch_size,self.max_length,4*self.tone_emb_size)

		# vowel = vowel[:,:,0].contiguous()
		vowel = self.vowel_embed(vowel)


		c_0 = self.init_hidden()
		h_0 = self.init_hidden()

		# print(pos.size())
		# print(pos_feat.size())
		# dep = self.dep_lemb(dep)
		emb = self.emb_l1(emb)
		feat_h_0 = torch.cat((emb,feat,pos,pos_feat),dim=2)
		feat_h_n, (_,_) = self.feat_lstm(feat_h_0,(h_0,c_0))
		feat_h = self.feat_l1(feat_h_n)
		feat_h = self.tanh(feat_h)
		feat_h = self.feat_l2(feat_h)

		c_0 = self.init_phrase_hidden()
		h_0 = self.init_phrase_hidden()

		ph_h_0 = torch.cat((feat,tone,cons,vowel,phrase),dim=2)
		# ph_h_0 = torch.cat((tone,cons,vowel),dim=2)
		ph_h_n, (_,_) = self.phrase_lstm(ph_h_0,(h_0,c_0))
		ph_h = self.phrase_l1(ph_h_n)
		ph_h = self.relu(ph_h)
		ph_h = self.phrase_l2(ph_h)

		h = feat_h+ph_h

		# delta,delta_length = self.get_f0_delta(h)
		# delta,delta_length = self.get_self_f0_delta(h)
		# delta,delta_length = self.get_mean_delta(h)
		# h = torch.cat((h,delta),dim=2)
		

		# h = h.view(self.batch_size,self.max_length*self.f0_dim)
		# h = h.view(self.batch_size,self.max_length*(self.f0_dim+delta_length))
		################################################################################
		# feat_h = feat_h.view(self.batch_size,self.max_length*self.f0_dim)
		# ph_h = ph_h.view(self.batch_size,self.max_length*self.f0_dim)
		# return h,feat_h,ph_h
		################################################################################
		return h


class TEST_MODEL(nn.Module):
	def __init__(self,emb_size,pos_emb_size,tone_emb_size,
		cons_num,vowel_num,pretone_num,tone_num,postone_num,feat_size,phrase_num,dep_num,voc_size,pos_num,pos_feat_num,
		lstm_hidden_size,f0_dim,linear_h1):
		super(TEST_MODEL, self).__init__()
		self.emb_size = emb_size
		self.feat_size = feat_size
		self.pos_emb_size = pos_emb_size
		self.pos_emb_length = 3##how many pos emb per sample (pre,current,post)
		self.tone_emb_size = tone_emb_size
		self.phrase_num = phrase_num
		self.dep_num = dep_num
		self.dep_lemb_size = 20
		self.emb_l_size = 10
		self.grad_emb_size = 10

		self.pretone_num = pretone_num
		self.tone_num = tone_num
		self.postone_num = postone_num
		self.cons_num = cons_num
		self.vowel_num = vowel_num

		self.lstm_hidden_size = lstm_hidden_size
		self.f0_dim = f0_dim
		# self.f0_dim = 1
		self.linear_h1 = linear_h1
		self.voc_size = voc_size
		self.pos_num = pos_num
		self.pos_feat_num = pos_feat_num
		self.batch_size = -1
		self.max_length = -1

		self.phrase_hidden_size = self.lstm_hidden_size
		self.phrase_linear_size = self.linear_h1

		self.grad_embed = nn.Embedding(self.voc_size, self.grad_emb_size,padding_idx=0)
		init.uniform(self.grad_embed.weight,a=-0.01,b=0.01)

		self.embed = self.get_embedding("./lstm_data/pretrain_emb",self.voc_size,self.emb_size)

		self.pos_embed = nn.Embedding(self.pos_num, self.pos_emb_size,padding_idx=0)
		init.uniform(self.pos_embed.weight,a=-0.01,b=0.01)

		self.tone_embed = nn.Embedding(self.tone_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.tone_embed.weight,a=-0.01,b=0.01)

		self.cons_embed = nn.Embedding(self.cons_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.cons_embed.weight,a=-0.01,b=0.01)
		self.vowel_embed = nn.Embedding(self.vowel_num, self.tone_emb_size,padding_idx=0)
		init.uniform(self.vowel_embed.weight,a=-0.01,b=0.01)
		# self.vowel_ch_embed = nn.Embedding(self.vowel_ch_num, self.tone_emb_size,padding_idx=0)
		# init.uniform(self.vowel_ch_embed.weight,a=-0.01,b=0.01)

		self.non_linear = nn.ReLU()
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()

		self.all_feat_length = self.emb_l_size+self.feat_size+self.pos_emb_length*self.pos_emb_size+self.pos_feat_num+\
			3*self.tone_emb_size+self.phrase_num


		##LSTM
		self.lstm_layer = 1
		self.bidirectional_flag = True
		self.direction = 2 if self.bidirectional_flag else 1
		self.feat_lstm = nn.LSTM(self.all_feat_length,self.lstm_hidden_size,
			num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)

		self.lstm_l = nn.Sequential(
			nn.Linear(self.lstm_hidden_size*self.direction,self.linear_h1),
			nn.ReLU(),
			nn.Linear(self.linear_h1,self.f0_dim)
			)

		self.mlp = nn.Sequential(
			nn.Linear(self.all_feat_length,300),
			nn.Sigmoid(),
			nn.Linear(300,200),
			nn.Sigmoid(),
			nn.Linear(200,100),
			nn.ReLU(),
			nn.Linear(100,self.f0_dim)
			)

		self.ngram_side = 1
		self.ngram_mlp = nn.Sequential(
			nn.Linear(self.all_feat_length*(2*self.ngram_side+1),400),
			nn.Sigmoid(),
			nn.Linear(400,200),
			nn.Sigmoid(),
			nn.Linear(200,100),
			nn.ReLU(),
			nn.Linear(100,self.f0_dim)
			)

		self.emb_l1 = nn.Linear(self.emb_size,self.emb_l_size)

	def get_ngram(self,data,n):
		batch_size,max_length,feat_size = data.size()
		ngram = None
		data_list = []
		for i in range(n):
			if cuda_flag:
				tmp = Variable(torch.zeros(batch_size,max_length,feat_size).cuda(async=True))
			else:
				tmp = Variable(torch.zeros(batch_size,max_length,feat_size))
			tmp[:,n-i:,:] = data[:,0:-i-1,:]
			data_list.append(tmp)
		data_list.append(tmp)
		for i in range(n):
			if cuda_flag:
				tmp = Variable(torch.zeros(batch_size,max_length,feat_size).cuda(async=True))
			else:
				tmp = Variable(torch.zeros(batch_size,max_length,feat_size))
			tmp[:,0:-i-1,:] = data[:,i+1:,:]
			data_list.append(tmp)
		ngram = torch.cat(data_list,dim=2)
		return ngram



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

	def get_embedding(self,emb_file,voc_size,emb_size):
		arr = np.loadtxt(emb_file)
		embed = nn.Embedding(voc_size, emb_size)
		embed.weight.data.copy_(torch.from_numpy(arr))
		embed.weight.requires_grad = False
		return embed

	def forward(self,sents,pos,pos_feat,cons,vowel,pretone,tone,postone,feat,phrase,dep,sent_length):
		self.batch_size,self.max_length = sents.size()
		emb = self.embed(sents)
		grad_emb = self.grad_embed(sents)

		pos = self.pos_embed(pos.view(self.batch_size,self.max_length*self.pos_emb_length))
		pos = pos.view(self.batch_size,self.max_length,self.pos_emb_length*self.pos_emb_size)

		tone = self.tone_embed(tone)
		cons = self.cons_embed(cons)

		# vowel_ch = vowel[:,:,1:].contiguous()
		# vowel_ch = self.vowel_ch_embed(vowel_ch.view(self.batch_size,self.max_length*4))
		# vowel_ch = vowel_ch.view(self.batch_size,self.max_length,4*self.tone_emb_size)

		# vowel = vowel[:,:,0].contiguous()
		vowel = self.vowel_embed(vowel)

		emb = self.emb_l1(emb)

		all_feat = torch.cat((emb,feat,pos,pos_feat,tone,cons,vowel,phrase),dim=2)
		y = self.mlp(all_feat)

		# c_0 = self.init_hidden()
		# h_0 = self.init_hidden()
		# h_n, (_,_) = self.feat_lstm(all_feat,(h_0,c_0))
		# y = self.lstm_l(h_n)

		# y = self.ngram_mlp(self.get_ngram(all_feat,self.ngram_side))
		return y

class NGram(nn.Module):
	def __init__(self,in_dim,win_size,out_dim):
		super(NGram, self).__init__()
		self.conv1 = nn.Conv1d(1,out_dim,kernel_size=win_size*in_dim,stride=in_dim,padding=(win_size-1)/2*in_dim)
		self.batch_size = -1
		self.length = -1
		self.win_size = win_size
		self.in_dim = in_dim
		self.out_dim = out_dim
	def forward(self,in_data):
		self.batch_size,self.length,_ = in_data.size()
		in_data = in_data.view(self.batch_size,1,self.length*self.in_dim)
		out_data = self.conv1(in_data).permute(0,2,1)
		return out_data


def Train(train_emb,train_pos,train_pos_feat,train_cons,train_vowel,train_pretone,train_tone,train_postone,train_feat,train_phrase,train_dep,train_f0,train_len,
	val_emb,val_pos,val_pos_feat,val_cons,val_vowel,val_pretone,val_tone,val_postone,val_feat,val_phrase,val_dep,val_f0,val_len,
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
				train_pos_feat_batch = Variable(train_pos_feat[i].cuda(async=True))
				train_cons_batch = Variable(train_cons[i].cuda(async=True))
				train_vowel_batch = Variable(train_vowel[i].cuda(async=True))
				train_tone_batch = Variable(train_tone[i].cuda(async=True))
				train_pretone_batch = Variable(train_pretone[i].cuda(async=True))
				train_postone_batch = Variable(train_postone[i].cuda(async=True))
				train_feat_batch = Variable(train_feat[i].cuda(async=True))
				train_f0_batch = Variable(train_f0[i].cuda(async=True))
				train_len_batch = Variable(train_len[i].cuda(async=True))
				train_phrase_batch = Variable(train_phrase[i].cuda(async=True))
				train_dep_batch = Variable(train_dep[i].cuda(async=True))
			else:
				train_emb_batch = Variable(train_emb[i])
				train_pos_batch = Variable(train_pos[i])
				train_pos_feat_batch = Variable(train_pos_feat[i])
				train_cons_batch = Variable(train_cons[i])
				train_vowel_batch = Variable(train_vowel[i])
				train_tone_batch = Variable(train_tone[i])
				train_pretone_batch = Variable(train_pretone[i])
				train_postone_batch = Variable(train_postone[i])
				train_feat_batch = Variable(train_feat[i])
				train_f0_batch = Variable(train_f0[i])
				train_len_batch = Variable(train_len[i])
				train_phrase_batch = Variable(train_phrase[i])
				train_dep_batch = Variable(train_dep[i])
			###########################################################


			optimizer.zero_grad()
			outputs = model(train_emb_batch,train_pos_batch,train_pos_feat_batch,train_cons_batch,train_vowel_batch,
				train_pretone_batch,train_tone_batch,train_postone_batch,train_feat_batch,train_phrase_batch,train_dep_batch,train_len_batch)
			
			# delta,delta_length = model.get_self_f0_delta(train_f0_batch)
			# delta,delta_length = model.get_f0_delta(train_f0_batch)
			# delta,delta_length = model.get_mean_delta(train_f0_batch)
			# train_f0_batch = torch.cat((train_f0_batch,delta),dim=2)

			loss = LF(outputs,train_f0_batch)
			loss.backward()
			optimizer.step()
			loss_val += loss.data[0]
		if (epoch+1)%1==0:
			print("Epoch "+str(epoch))
			print("train loss: "+str(loss_val/len(train_emb)))
			val_loss,result = Validate(model,val_emb,val_pos,val_pos_feat,val_cons,val_vowel,val_pretone,val_tone,val_postone,val_feat,val_phrase,val_dep,val_f0,val_len)
			print("val loss: "+str(val_loss))
			if val_loss<min_loss:
				torch.save(model,"./my_best_model.model")
				min_loss = val_loss
		if (epoch+1)%decay_step==0:
			learning_rate *= decay_rate
			# print(model)
			# print(str(optimizer.param_groups).decode("utf-8"))
			for param_group in optimizer.param_groups:
				# print(param_group.keys())
				if param_group["my_name"] not in ["embed.weight","pos_embed.weight","cons_embed.weight","vowel_embed.weight"]:
					param_group['lr'] = learning_rate
				else:
					param_group['lr'] = learning_rate
					# if epoch>=1:
					# 	# print("find embed.weight")
					# 	param_group['lr'] = learning_rate
					# else:
					# 	param_group['lr'] = 10*learning_rate
			print("#####################################")
			print("learning rate: "+str(learning_rate))
			print("#####################################")
		print("time: "+str(time.time()-start_time))
		print("#####################################")

def Validate(model,val_emb,val_pos,val_pos_feat,val_cons,val_vowel,val_pretone,val_tone,val_postone,val_feat,val_phrase,val_dep,val_f0,val_len,save_prediction=""):
	model.eval()
	val_f0_shape = val_f0.size()
	batch_size = val_f0_shape[0]

	###########################################################
	#GPU OPTION
	###########################################################
	if cuda_flag:
		result = model(Variable(val_emb.cuda(async=True)),Variable(val_pos.cuda(async=True)),Variable(val_pos_feat.cuda(async=True)),
			Variable(val_cons.cuda(async=True)),Variable(val_vowel.cuda(async=True)),Variable(val_pretone.cuda(async=True)),
			Variable(val_tone.cuda(async=True)),Variable(val_postone.cuda(async=True)),Variable(val_feat.cuda(async=True)),
			Variable(val_phrase.cuda(async=True)),Variable(val_dep.cuda(async=True)),
			Variable(val_len.cuda(async=True)))
		# result = result.data.cpu().numpy().reshape((batch_size,model.max_length,model.f0_dim))
		result = result.data.cpu().numpy()[:,:,0:model.f0_dim]
		# val_f0 = val_f0.cpu().numpy().reshape((batch_size,model.max_length,model.f0_dim))
		val_f0 = val_f0.cpu().numpy()
	else:
		result = model(Variable(val_emb),Variable(val_pos),Variable(val_pos_feat),Variable(val_cons),Variable(val_vowel),
			Variable(val_pretone),Variable(val_tone),Variable(val_postone),
			Variable(val_feat),Variable(val_phrase),Variable(val_dep),Variable(val_len))
		
		# result = result.data.numpy().reshape((batch_size,model.max_length,model.f0_dim))
		result = result.data.numpy()[:,:,0:model.f0_dim]
		# val_f0 = val_f0.numpy().reshape((batch_size,model.max_length,model.f0_dim))
		val_f0 = val_f0.numpy()
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

	###########################################################
	# prediction = prediction[:,0:-1]
	# true_f0 = true_f0[:,0:-1]
	###########################################################

	loss_arr = np.sqrt(np.square(prediction-true_f0).mean(axis=1))
	np.random.shuffle(loss_arr)
	print("first half loss:"+str(loss_arr[0:int(len(loss_arr)/2)].mean()))
	print("second half loss:"+str(loss_arr[int(len(loss_arr)/2):].mean()))
	# print("mean abs:")
	# print(np.abs(prediction.mean(axis=1)-true_f0.mean(axis=1)).mean())
	# print("std abs:")
	# print(np.abs(prediction.std(axis=1)-true_f0.std(axis=1)).mean())
	loss = loss_arr.mean()

	if save_prediction!="":
		np.savetxt(save_prediction,prediction,delimiter=" ",fmt="%.3f")

	return loss,result.reshape((val_f0_shape))




