import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from scipy.fftpack import idct, dct
import argparse
import time
class EncodeFeature():
	def __init__(self,desc):
		self.encode_dic = []
		with open(desc) as f:
			cont = f.readlines()
			cont = [line.strip()[1:-1] for line in cont]
			cont.pop(0)
			cont.pop(0)
			cont.pop(-1)
		for feat in cont:
			feat = feat.split(" ")
			if feat[-1]=="float":
				self.encode_dic.append({})
			else:
				tmp_dic = {}
				for i in range(1,len(feat)):
					tmp_dic[feat[i]] = i-1
				self.encode_dic.append(tmp_dic)
		self.not_one_hot_flag = []
		count = 0
		for i in range(len(self.encode_dic)):
			if len(self.encode_dic[i])==0:
				self.not_one_hot_flag.append(count)
				count += 1
			else:
				count += len(self.encode_dic[i])
	def encode(self,feat):
		feat = feat.split(" ")
		feat.pop(0)
		feat.pop(0)
		new_feat = []
		# print(feat)
		# print(self.encode_dic)
		assert len(feat)==len(self.encode_dic)
		for i in range(len(feat)):
			if len(self.encode_dic[i])==0:
				new_feat.append(feat[i])
			else:
				one_hot = [0 for j in range(len(self.encode_dic[i]))]
				# print(self.encode_dic[i])
				# print(feat[i])
				one_hot[self.encode_dic[i][feat[i]]] = 1
				new_feat.append(" ".join([str(val) for val in one_hot]))
		new_feat = " ".join(new_feat)
		return new_feat




class Net(nn.Module):
	def __init__(self,feature_number,out_size):
		super(Net, self).__init__()
		self.hidden_unit = 100
		self.feature_number = feature_number
		self.l1 = nn.Linear(feature_number,self.hidden_unit)
		# self.l1.weight.data.uniform_(-0.1, 0.1)
		# self.l1.bias.data.uniform_(-1, 1)
		# self.init(self.l1)

		self.l2 = nn.Linear(self.hidden_unit,self.hidden_unit)
		# self.init(self.l2)

		self.l3 = nn.Linear(self.hidden_unit,out_size)
		# self.init(self.l3)

		# self.l2.weight.data.uniform_(-0.1, 0.1)
		# self.l2.bias.data.uniform_(-1, 1)

		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.drop = nn.Dropout(0)

	def init(self,layer):
		layer.weight.data.uniform_(-1, 1)
		layer.bias.data.uniform_(-1, 1)


	def forward(self,x):
		x = self.l1(x)
		
		# x = self.sigmoid(x)
		x = self.tanh(x)
		# x = self.relu(x)
		x = self.l2(self.drop(x))

		# x = self.sigmoid(x)
		x = self.tanh(x)
		# x = self.relu(x)
		x = self.l3(self.drop(x))

		return x
def convert_feature(feat_file,f0_file,encode_feature,out_file):
	with open(feat_file) as featf, open(f0_file) as f0f, open(out_file,"w+") as outf:
		feat = featf.readlines()
		f0 = f0f.readlines()
		assert len(feat)==len(f0)
		for i in range(len(f0)):
			outf.write(f0[i].strip()+" "+encode_feature.encode(feat[i].strip())+"\n")

	##Normalize not one hot vector
	# not_one_hot = np.array(encode_feature.not_one_hot_flag)+10
	# data = np.loadtxt(out_file,delimiter=" ")
	# not_one_hot_data = data[:,not_one_hot]
	# data[:,not_one_hot] = (not_one_hot_data-not_one_hot_data.mean(axis=0))/(not_one_hot_data.std(axis=0)+0.0001)
	# np.savetxt(out_file,data,delimiter=" ",fmt="%.3f")

def convert_feature_old(feat_file,f0_file,out_file):
	f0 = np.loadtxt(f0_file,delimiter=" ")
	out_file = open(out_file,"w+")
	r = 0
	with open(feat_file) as f:
		for line in f:
			line = line.strip().split(" ")
			for i in range(len(line)):
				if line[i]=="NONE":
					line[i] = "-1"
				elif line[i]=="Accented":
					line[i] = "1"
			line = [str(val) for val in list(f0[r])]+line[2:]
			out_file.write(" ".join(line)+"\n")
			r += 1
def Validate(model,val_data,val_label,dct_flag):
	model.eval()
	val_label_shape = val_label.size()
	result = model(Variable(val_data)).data.numpy()
	# print(result.shape)
	if dct_flag:
		result = idct(result,axis=1)/(2*result.shape[1])
	result = result.flatten()
	val_label = val_label.numpy().flatten()
	loss = np.sqrt(np.square(val_label-result).mean())
	return loss,result.reshape((val_label_shape))

def concat_seq_feature(in_file,map_file,out_file):
	data = np.loadtxt(in_file,delimiter=" ")
	new_data = np.zeros((data.shape[0],data.shape[1]*2-10))
	with open(map_file) as mf:
		pre_data = ""
		cont = mf.readlines()
		for i in range(len(cont)):
			data_name = cont[i].strip().split(" ")[0]
			new_data[i,0:data.shape[1]] = data[i]
			if data_name == pre_data:
				new_data[i,data.shape[1]:] = data[i-1,10:]
			pre_data = data_name
	np.savetxt(out_file,new_data,delimiter=" ",fmt="%.3f")





def Train(train_data,train_label,val_data,val_label,dct_flag):
	model = Net(train_data.size()[1],train_label.size()[1])
	# optimizer = optim.SGD(model.parameters(), lr=0.01)
	learning_rate = 0.01
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	decay_step = 10
	decay_rate = 1
	
	LF = nn.MSELoss()
	min_loss = 10000
	for epoch in range(50):
		start_time = time.time()
		loss_val = 0
		for i in range(len(train_data)):
			# if i==6:
			# 	return
			train_data_batch = Variable(train_data[i])
			train_label_batch = Variable(train_label[i])

			optimizer.zero_grad()
			outputs = model(train_data_batch)
			# print(outputs.size())
			# print(train_label_batch.size())
			loss = LF(outputs,train_label_batch)
			loss.backward()
			optimizer.step()
			loss_val += loss.data[0]
		if (epoch+1)%1==0:
			print("Epoch "+str(epoch))
			print("train loss: "+str(loss_val/len(train_data)))
			val_loss,result = Validate(model,val_data,val_label,dct_flag)
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

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode')
	parser.add_argument('--desc_file', dest='desc_file')
	parser.add_argument('--train_data', dest='train_data')
	parser.add_argument('--train_label', dest='train_label')
	parser.add_argument('--test_data', dest='test_data')
	parser.add_argument('--test_label', dest='test_label')
	args = parser.parse_args()
	mode = args.mode
	if mode=="how_to_run":
		print("python train_f0.py"+
			" --mode train"+
			" --desc_file ./feature_desc"+
			" --train_data ./data/train/dct_0"+
			" --train_label ./data/train_data_f0_vector"+
			" --test_data ./data/dev/dct_0"+
			" --test_label ./data/dev_data_f0_vector")
	elif mode=="train":
		desc_file = args.desc_file
		train_data = args.train_data
		train_label = args.train_label
		test_data = args.test_data
		test_label = args.test_label

		dct_flag = False
		mean = False
		# encode_feature = EncodeFeature("./feature_desc")
		encode_feature = EncodeFeature(desc_file)

		# convert_feature("./data/train/dct_0","./data/train_data_f0_vector",encode_feature,"./train_data_f0")
		# convert_feature("./data/dev/dct_0","./data/dev_data_f0_vector",encode_feature,"./dev_data_f0")
		convert_feature(train_data,train_label,encode_feature,"./train_data_f0")
		convert_feature(test_data,test_label,encode_feature,"./dev_data_f0")

		# concat_seq_feature("./train_data_f0","./data/train_syllable_map","./train_data_f0")
		# concat_seq_feature("./dev_data_f0","./data/dev_syllable_map","./dev_data_f0")
		################################################################################
		train_ratio = 0.8
		data = np.loadtxt("./train_data_f0",delimiter=" ")
		np.random.shuffle(data)
		train_data = data[0:int(train_ratio*len(data)),10:]
		train_data = (train_data-train_data.mean(axis=0))/(train_data.std(axis=0)+0.0001)
		train_label = data[0:int(train_ratio*len(data)),0:10]
		if dct_flag:
			train_label = dct(train_label,axis=1)
		if mean:
			train_label = train_label.mean(axis=1).reshape((-1,1))

		val_data = data[int(train_ratio*len(data)):,10:]
		val_data = (val_data-val_data.mean(axis=0))/(val_data.std(axis=0)+0.0001)
		val_label = data[int(train_ratio*len(data)):,0:10]
		if mean:
			val_label = val_label.mean(axis=1).reshape((-1,1))

		Train(torch.FloatTensor(train_data),torch.FloatTensor(train_label),torch.FloatTensor(val_data),torch.FloatTensor(val_label),dct_flag)
		################################################################################


		################################################################################
		data = np.loadtxt("./dev_data_f0",delimiter=" ")
		test_data = data[:,10:]
		test_data = (test_data-test_data.mean(axis=0))/(test_data.std(axis=0)+0.0001)
		test_label = data[:,0:10]
		if mean:
			test_label = test_label.mean(axis=1).reshape((-1,1))
		model = torch.load("./my_best_model_.model")
		test_loss,result = Validate(model,torch.FloatTensor(test_data),torch.FloatTensor(test_label),dct_flag)
		print("test loss: "+str(test_loss))
		np.savetxt("prediction",result,fmt="%.3f")
		################################################################################
