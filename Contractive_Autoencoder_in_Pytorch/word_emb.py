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
import time
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import random
def show_heatmap(arr):
	plt.imshow(arr, cmap='hot', interpolation='nearest')
	plt.show()

class CAE(nn.Module):
	def __init__(self,win_size,vocab_size):
		super(CAE, self).__init__()
		self.h_size = 100
		self.vocab_size = vocab_size
		self.win_size = win_size
		self.emb_size = 200
		self.batch_size = -1

		self.embed = nn.Embedding(self.vocab_size, self.emb_size,padding_idx=0)
		# init.uniform(self.embed.weight,a=-0.01,b=0.01)
		# self.embed = self.get_embedding()
		self.l1 = nn.Linear((self.win_size-1)*self.emb_size,self.h_size)
		self.l2 = nn.Linear(self.h_size,self.vocab_size)

		self.embl = nn.Linear(300,self.emb_size)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax()


	def forward(self,word):
		self.batch_size = word.size()[0]
		emb = self.embed(word.view(-1,1))
		# emb = self.embl(emb)
		emb = emb.view(self.batch_size,(self.win_size-1)*self.emb_size)
		emb = self.l1(emb)
		# emb = emb.view(self.batch_size,self.win_size-1,self.h_size)
		# emb = torch.sum(emb,dim=1)
		emb = self.tanh(emb)
		emb = self.l2(emb)
		prob = self.softmax(emb)
		return prob

	def get_embedding(self,emb_file="../lstm_data/pretrain_emb",voc_size=3601,emb_size=300):
		arr = np.loadtxt(emb_file)
		embed = nn.Embedding(voc_size, emb_size)
		embed.weight.data.copy_(torch.from_numpy(arr))
		embed.weight.requires_grad = False
		return embed

def Validate(model,x,y):
	model.eval()
	prob = model(x)
	prob = prob.cpu().data.numpy()
	predict = np.argmax(prob,axis=1)
	y = y.cpu().data.numpy()
	print(predict[0:10])
	# print(y[0:10])
	count = 0
	for i in range(len(y)):
		if predict[i]==y[i]:
			count += 1
	return float(count)/len(y)

def get_data_label(in_dir,win_size):
	train_file = os.listdir(in_dir)
	train_data = []
	train_label = []
	padding = [0 for i in range(int((win_size-1)/2))]
	half_win = int((win_size-1)/2)
	for file in train_file:
		arr = np.loadtxt(in_dir+"/"+file)[:,81]
		arr = np.hstack(padding+[arr]+padding)
		for i in range(half_win,len(arr)-half_win):
			# if arr[i]<10 or arr[i]==2499:
			if arr[i]<10:
				if random.random()>0.3:
					continue
			train_data.append([arr[i-half_win:i],arr[i+1:i+1+half_win]])
			train_label.append(arr[i])
	train_data = np.array(train_data).reshape((-1,win_size-1)).astype(np.int16)
	train_label = np.array(train_label).reshape((-1,)).astype(np.int16)
	return train_data,train_label

if __name__=="__main__":
	if sys.argv[1]=="train":
		win_size = 5
		print("reading data...")
		train_data,train_label = get_data_label("../lstm_data/train",win_size)
		# print(train_data[0:100])
		test_data,test_label = get_data_label("../lstm_data/test",win_size)
		# print(test_label[0:100])
		vocab_size = 3601
		# print(test_data.shape)
		# print(test_label.shape)
		

		print(train_data.shape)
		train_data = train_data[0:56000]
		train_label = train_label[0:56000]

		batch_size = 50
		batch_num = int(56000/batch_size)
		train_data = train_data.reshape((batch_num,batch_size,win_size-1))
		train_label = train_label.reshape((batch_num,batch_size))

		train_data = torch.LongTensor(train_data.tolist())
		train_label = torch.LongTensor(train_label.tolist())

		test_data = torch.LongTensor(test_data.tolist())
		test_label = torch.LongTensor(test_label.tolist())

		test_data = Variable(test_data).cuda()
		test_label = Variable(test_label).cuda()


		ce_loss = nn.CrossEntropyLoss()
		model = CAE(win_size,vocab_size)

		torch.backends.cudnn.benchmark = True
		model.cuda()

		# model.load_state_dict(torch.load("best_model_con"))
		# optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr = 0.005,weight_decay=0)
		# print(len([param for param in model.parameters() if param.requires_grad]))
		optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr = 0.005,weight_decay=0)
		# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
		val_recons = None
		max_acc = 0
		beat_model = None
		# one_arr = torch.ones(batch_size).long()
		print("begin training...")
		for epoch in range(100):
			start_time = time.time()
			loss_val = 0
			for i in range(batch_num):
				train_data_batch = Variable(train_data[i]).cuda()
				# train_label_batch = torch.zeros(batch_size,vocab_size).long()
				# print(train_label[i].size())
				# print(train_label_batch.size())
				# train_label_batch.scatter_(1,train_label[i].view(-1,1),1)
				train_label_batch = Variable(train_label[i]).cuda()

				optimizer.zero_grad()

				prob = model(train_data_batch)

				# loss = loss_function(W,train_batch,recons_x,h,mse_loss)z
				loss = ce_loss(prob, train_label_batch)
				loss.backward()
				optimizer.step()

				loss_val += loss.data[0]

			for param_group in optimizer.param_groups:
				param_group["lr"] *= 0.99

			print("Epoch "+str(epoch))
			print("train loss: "+str(loss_val/batch_num))
			acc = Validate(model,test_data,test_label)
			print("Accuracy: "+str(acc))
			print("time: "+str(time.time()-start_time))
			print("############################################")

			if max_acc < acc:
				max_acc = acc
				best_model = copy.deepcopy(model)
		torch.save(best_model.state_dict(),"best_model")

	elif sys.argv[1]=="save_emb":
		model = CAE(3,3601)
		model.load_state_dict(torch.load("best_model"))
		emb = model.state_dict()['embed.weight'].numpy()
		np.savetxt("pretrain_emb",emb,delimiter=" ")
		print(emb.shape)
		

	