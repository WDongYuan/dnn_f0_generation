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

class Seq2Seq(nn.Module):
	def __init__(self,emb_size,pos_emb_size,tone_emb_size,
		cons_num,vowel_num,pretone_num,tone_num,postone_num,feat_size,phrase_num,dep_num,voc_size,pos_num,pos_feat_num,
		lstm_hidden_size,f0_dim,linear_h1):
		super(Seq2Seq, self).__init__()
		self.emb_size = emb_size
		self.feat_size = feat_size
		self.pos_emb_size = pos_emb_size
		self.pos_emb_length = 3##how many pos emb per sample (pre,current,post)
		self.tone_emb_size = tone_emb_size
		self.phrase_num = phrase_num
		self.dep_num = dep_num
		self.dep_lemb_size = 20
		self.emb_l_size = 100
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
		self.feat_lstm = nn.LSTM(self.f0_dim+self.emb_l_size+self.feat_size+self.pos_emb_length*self.pos_emb_size+self.pos_feat_num,self.lstm_hidden_size,
			num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)
		self.phrase_lstm = nn.LSTM(self.phrase_num+3*self.tone_emb_size, self.phrase_hidden_size,
			num_layers=self.lstm_layer,bidirectional=self.bidirectional_flag,batch_first=True)


		self.non_linear = nn.ReLU()
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.drop = nn.Dropout(0)

		self.feat_l1 = nn.Linear(self.lstm_hidden_size*self.direction,self.linear_h1)
		self.linear_init(self.feat_l1)
		self.feat_l2 = nn.Linear(self.linear_h1,self.f0_dim)
		self.linear_init(self.feat_l2)

		self.emb_l1 = nn.Linear(self.emb_size,self.emb_l_size)


	def linear_init(self,layer,lower=-1,upper=1):
		layer.weight.data.uniform_(lower, upper)
		layer.bias.data.uniform_(lower, upper)
	def init_hidden(self,batch_size):
		direction = 2 if self.bidirectional_flag else 1
		###########################################################
		#GPU OPTION
		###########################################################
		if cuda_flag:
			return Variable(torch.rand(self.lstm_layer*direction,batch_size,self.lstm_hidden_size).cuda(async=True))
		else:
			return Variable(torch.rand(self.lstm_layer*direction,batch_size,self.lstm_hidden_size))
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

	def get_embedding(self,emb_file,voc_size,emb_size):
		arr = np.loadtxt(emb_file)
		embed = nn.Embedding(voc_size, emb_size)
		embed.weight.data.copy_(torch.from_numpy(arr))
		embed.weight.requires_grad = False
		return embed


	def forward(self,sents,pos,pos_feat,cons,vowel,pretone,tone,postone,feat,phrase,dep,sent_length,pre_f0,h_0,c_0):
		self.batch_size,self.max_length = sents.size()
		emb = self.embed(sents)
		grad_emb = self.grad_embed(sents)
		pos = pos.contiguous()
		pos = self.pos_embed(pos.view(self.batch_size,self.max_length*self.pos_emb_length))
		pos = pos.view(self.batch_size,self.max_length,self.pos_emb_length*self.pos_emb_size)
		# pretone = self.pretone_embed(pretone)
		tone = self.tone_embed(tone)
		# postone = self.postone_embed(postone)
		cons = self.cons_embed(cons)
		vowel = self.vowel_embed(vowel)

		emb = self.emb_l1(emb)
		feat_h_0 = torch.cat((emb,feat,pos,pos_feat,pre_f0),dim=2)
		h_0 = self.drop(h_0)
		c_0 = self.drop(c_0)
		feat_h_n, (h_t,c_t) = self.feat_lstm(feat_h_0,(h_0,c_0))
		feat_h_n = self.drop(feat_h_n)
		feat_h = self.feat_l1(feat_h_n)
		feat_h = self.relu(feat_h)
		feat_h = self.drop(feat_h)
		feat_h = self.feat_l2(feat_h)

		h = feat_h

		h = h.view(self.batch_size,self.max_length,self.f0_dim)
		return h,h_t,c_t




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
	pre_f0 = create_training_f0(train_f0.view(train_f0.size()[0],train_f0.size()[1],-1,model.f0_dim))
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
				pre_f0_batch = Variable(pre_f0[i].cuda(async=True))
				pass
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
				pre_f0_batch = Variable(pre_f0[i])
				pass
			###########################################################


			optimizer.zero_grad()
			batch_size,max_length = train_emb[i].size()
			h_0 = model.init_hidden(batch_size)
			c_0 = model.init_hidden(batch_size)
			
			# pre_f0 = None
			# if cuda_flag:
			# 	pre_f0 = Variable(torch.zeros(batch_size,1,model.f0_dim).cuda(async=True))
			# else:
			# 	pre_f0 = Variable(torch.zeros(batch_size,1,model.f0_dim))
			# outputs = Variable(torch.zeros(batch_size,max_length,model.f0_dim).cuda(async=True))
			# for l in range(max_length):
			# 	tmp_result,h_0,c_0 = model(
			# 		train_emb_batch[:,l:l+1].cuda(),
			# 		train_pos_batch[:,l:l+1].cuda(),
			# 		train_pos_feat_batch[:,l:l+1].cuda(),
			# 		train_cons_batch[:,l:l+1].cuda(),
			# 		train_vowel_batch[:,l:l+1].cuda(),
			# 		train_pretone_batch[:,l:l+1].cuda(),
			# 		train_tone_batch[:,l:l+1].cuda(),
			# 		train_postone_batch[:,l:l+1].cuda(),
			# 		train_feat_batch[:,l:l+1].cuda(),
			# 		train_phrase_batch[:,l:l+1].cuda(),
			# 		train_dep_batch[:,l:l+1].cuda(),
			# 		train_len_batch.cuda(),
			# 		pre_f0,h_0,c_0)
			# 	pre_f0 = tmp_result
			# 	outputs[:,l:l+1] = tmp_result
			outputs,h_0,c_0 = model(
					train_emb_batch,
					train_pos_batch,
					train_pos_feat_batch,
					train_cons_batch,
					train_vowel_batch,
					train_pretone_batch,
					train_tone_batch,
					train_postone_batch,
					train_feat_batch,
					train_phrase_batch,
					train_dep_batch,
					train_len_batch,
					pre_f0,h_0,c_0)

			outputs = outputs.view(batch_size,-1)
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

def create_training_f0(in_f0):
	in_f0 = in_f0.numpy()
	f0 = np.zeros(in_f0.shape)
	f0[:,:,1:,:] = in_f0[:,:,0:-1,:]
	return torch.FloatTensor(f0.tolist())

def Validate(model,val_emb,val_pos,val_pos_feat,val_cons,val_vowel,val_pretone,val_tone,val_postone,val_feat,val_phrase,val_dep,val_f0,val_len,save_prediction=""):
	model.eval()
	val_f0_shape = val_f0.size()
	batch_size,max_length = val_emb.size()
	result = np.zeros((batch_size,max_length,model.f0_dim))

	h_0 = model.init_hidden(batch_size)
	c_0 = model.init_hidden(batch_size)
	pre_f0 = None
	if cuda_flag:
		pre_f0 = Variable(torch.zeros(batch_size,1,model.f0_dim).cuda(async=True))
		val_f0 = val_f0.cpu().numpy().reshape((batch_size,max_length,model.f0_dim))
	else:
		pre_f0 = Variable(torch.zeros(batch_size,1,model.f0_dim))
		val_f0 = val_f0.numpy().reshape((batch_size,max_length,model.f0_dim))

	for i in range(max_length):
		###########################################################
		#GPU OPTION
		###########################################################
		if cuda_flag:
			output,h_0,c_0 = model(Variable(val_emb[:,i:i+1].contiguous().cuda(async=True)),
				Variable(val_pos[:,i:i+1].contiguous().cuda(async=True)),
				Variable(val_pos_feat[:,i:i+1].contiguous().cuda(async=True)),
				Variable(val_cons[:,i:i+1].contiguous().cuda(async=True)),
				Variable(val_vowel[:,i:i+1].contiguous().cuda(async=True)),
				Variable(val_pretone[:,i:i+1].contiguous().cuda(async=True)),
				Variable(val_tone[:,i:i+1].contiguous().cuda(async=True)),
				Variable(val_postone[:,i:i+1].contiguous().cuda(async=True)),
				Variable(val_feat[:,i:i+1].contiguous().cuda(async=True)),
				Variable(val_phrase[:,i:i+1].contiguous().cuda(async=True)),
				Variable(val_dep[:,i:i+1].contiguous().cuda(async=True)),
				Variable(val_len.contiguous().cuda(async=True))
				,pre_f0,h_0,c_0)
			# result,h_0,c_0 = model(Variable(val_emb.cuda(async=True)),Variable(val_pos.cuda(async=True)),Variable(val_pos_feat.cuda(async=True)),
			# 	Variable(val_cons.cuda(async=True)),Variable(val_vowel.cuda(async=True)),Variable(val_pretone.cuda(async=True)),
			# 	Variable(val_tone.cuda(async=True)),Variable(val_postone.cuda(async=True)),Variable(val_feat.cuda(async=True)),
			# 	Variable(val_phrase.cuda(async=True)),Variable(val_dep.cuda(async=True)),
			# 	Variable(val_len.cuda(async=True)))
			result[:,i:i+1] = output.cpu().data.numpy()
			pre_f0 = output
		else:
			output,h_0,c_0= model(Variable(val_emb[:,i:i+1].contiguous()),
				Variable(val_pos[:,i:i+1].contiguous()),
				Variable(val_pos_feat[:,i:i+1].contiguous()),
				Variable(val_cons[:,i:i+1].contiguous()),
				Variable(val_vowel[:,i:i+1].contiguous()),
				Variable(val_pretone[:,i:i+1].contiguous()),
				Variable(val_tone[:,i:i+1].contiguous()),
				Variable(val_postone[:,i:i+1].contiguous()),
				Variable(val_feat[:,i:i+1].contiguous()),
				Variable(val_phrase[:,i:i+1].contiguous()),
				Variable(val_dep[:,i:i+1].contiguous()),
				Variable(val_len.contiguous()),
				pre_f0,h_0,c_0)
			# output,h_0,c_0= model(Variable(val_emb),Variable(val_pos),Variable(val_pos_feat),Variable(val_cons),Variable(val_vowel),
			# 	Variable(val_pretone),Variable(val_tone),Variable(val_postone),
			# 	Variable(val_feat),Variable(val_phrase),Variable(val_dep),Variable(val_len))
			# print(result.size())
			result[:,i:i+1] = output.data.numpy()
			pre_f0 = output
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

	loss = np.sqrt(np.square(prediction-true_f0).mean(axis=1)).mean()
	# print("abs:")
	# print(np.abs(prediction-true_f0).mean(axis=0))
	print("mean abs:")
	print(np.abs(prediction.mean(axis=1)-true_f0.mean(axis=1)).mean())
	print("std abs:")
	print(np.abs(prediction.std(axis=1)-true_f0.std(axis=1)).mean())

	if save_prediction!="":
		np.savetxt(save_prediction,prediction,delimiter=" ",fmt="%.3f")

	return loss,result.reshape((val_f0_shape))




