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
import os
from utils.data_processing import *
from model.mlp import MLP
from model import embedding_lstm
from model import feature_lstm
from model import emb_mean_std
from model import emb_att
from utils import config
# from model import expand_emb
###########################################################
#GPU OPTION
###########################################################
import torch.backends.cudnn as cudnn
###########################################################


def Validate(model,val_data,val_label,dct_flag=False):
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

def Train(train_data,train_label,val_data,val_label,\
	model,optimizer,learning_rate,decay_step,decay_rate,dct_flag=False):
	
	LF = nn.MSELoss()
	min_loss = 100000000
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
	parser.add_argument('--txt_file', dest='txt_file')
	parser.add_argument('--train_data', dest='train_data')
	parser.add_argument('--train_label', dest='train_label')
	parser.add_argument('--train_map', dest='train_map')
	parser.add_argument('--test_data', dest='test_data')
	parser.add_argument('--test_label', dest='test_label')
	parser.add_argument('--test_map', dest='test_map')
	args = parser.parse_args()
	mode = args.mode
	if mode=="how_to_run":
		print("python train.py"+
			" --mode train_mlp"+
			" --desc_file ./feature_desc"+
			" --train_data ./data/train/dct_0"+
			" --train_label ./data/train_data_f0_vector"+
			" --test_data ./data/dev/dct_0"+
			" --test_label ./data/dev_data_f0_vector")
		print("python train.py"+
			" --mode train_lstm"+
			" --desc_file ./feature_desc"+
			" --txt_file ../../txt.done.data"+
			" --train_data ./data/train/dct_0"+
			" --train_label ./data/train_data_f0_vector"+
			" --train_map ./data/train_syllable_map"+
			" --test_data ./data/dev/dct_0"+
			" --test_label ./data/dev_data_f0_vector"+
			" --test_map ./data/dev_syllable_map")
		print("python train.py"+
			" --mode train_emb_lstm/emb_lstm_predict"+
			" --desc_file ../mandarine/gen_f0/train_dev_data_vector/feature_desc_vector"+
			" --txt_file ../mandarine/txt.done.data-all"+
			" --train_data ../mandarine/gen_f0/train_dev_data_vector/train_data/dct_0"+
			" --train_label ../mandarine/gen_f0/train_dev_data_vector/train_data_f0_vector"+
			" --train_map ../mandarine/gen_f0/train_dev_data_vector/train_data/syllable_map"+
			" --test_data ../mandarine/gen_f0/train_dev_data_vector/dev_data/dct_0"+
			" --test_label ../mandarine/gen_f0/train_dev_data_vector/dev_data_f0_vector"+
			" --test_map ../mandarine/gen_f0/train_dev_data_vector/dev_data/syllable_map")
		
	elif mode=="train_mlp":
		desc_file = args.desc_file
		train_data = args.train_data
		train_label = args.train_label
		test_data = args.test_data
		test_label = args.test_label

		dct_flag = False
		mean = False

		encode_feature = EncodeFeature(desc_file)

		convert_feature(train_data,train_label,encode_feature,"./train_data_f0")
		convert_feature(test_data,test_label,encode_feature,"./dev_data_f0")

		# concat_seq_feature("./train_data_f0","./data/train_syllable_map","./train_data_f0")
		# concat_seq_feature("./dev_data_f0","./data/dev_syllable_map","./dev_data_f0")
		################################################################################
		train_ratio = 0.8
		train_data,train_label,val_data,val_label = create_train_val_data("./train_data_f0",train_ratio)
		if dct_flag:
			train_label = dct(train_label,axis=1)
		if mean:
			train_label = train_label.mean(axis=1).reshape((-1,1))
			val_label = val_label.mean(axis=1).reshape((-1,1))

		model = MLP(train_data.shape[1],train_label.shape[1])
		learning_rate = 0.01
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		decay_step = 10
		decay_rate = 0.3
		Train(torch.FloatTensor(train_data),torch.FloatTensor(train_label),torch.FloatTensor(val_data),torch.FloatTensor(val_label),\
			model,optimizer,learning_rate,decay_step,decay_rate,dct_flag)
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
	elif "train_emb" in mode or ("emb" in mode and "predict" in mode):
		desc_file = args.desc_file
		train_data = args.train_data
		train_label = args.train_label
		train_map = args.train_map
		test_data = args.test_data
		test_label = args.test_label
		test_map = args.test_map
		txt_file = args.txt_file

		normalize = True
		predict_val = "std"

		# encode_feature = EncodeFeature(desc_file)
		# convert_feature(train_data,train_label,encode_feature,"./train_data_f0")
		# convert_feature(test_data,test_label,encode_feature,"./test_data_f0")

		os.system("mkdir lstm_data")
		print("--->collect data according to the data name")
		# word_index = word2index(txt_file,config.voc_size)
		# collect_utt_data("./train_data_f0",train_map,"./lstm_data/train",txt_file,word_index)
		# collect_utt_data("./test_data_f0",test_map,"./lstm_data/test",txt_file,word_index)
		print("--->get the numpy data for training")
		train_f0,train_emb,train_len = get_f0_embedding("./lstm_data/train")
		test_f0,test_emb,test_len = get_f0_embedding("./lstm_data/test")
		if normalize:
			train_shape,train_mean,train_std = get_shape_mean_std(train_f0,train_len)
			test_shape,test_mean,test_std = get_shape_mean_std(test_f0,test_len)
			if predict_val=="shape":
				train_f0 = train_shape
				test_f0 = test_shape
			elif predict_val=="mean":
				train_f0 = train_mean
				test_f0 = test_mean
				config.f0_dim = 1
			elif predict_val=="std":
				train_f0 = train_std
				test_f0 = test_std
				config.f0_dim = 1

		batch_num = int(train_f0.shape[0]/config.batch_size)
		max_length = int(train_emb.shape[1])

		train_f0 = train_f0[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_emb = train_emb[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_len = train_len[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size))

		test_f0 = test_f0.reshape((len(test_f0),-1))
		test_emb = test_emb.reshape((len(test_emb),-1))
		test_len = test_len


		# print(np.sum(test_len))
		train_emb = torch.LongTensor(train_emb.tolist())
		train_f0 = torch.FloatTensor(train_f0.tolist())
		train_len = torch.LongTensor(train_len.tolist())
		test_emb = torch.LongTensor(test_emb.tolist())
		test_f0 = torch.FloatTensor(test_f0.tolist())
		test_len = torch.LongTensor(test_len.tolist())

		if "predict" in mode:
			print("predicting...")
			model = torch.load("./my_best_model_.model")
			if mode=="emb_lstm_predict":
				embedding_lstm.Validate(model,test_emb,test_f0,test_len,"./model_prediction")
			elif mode=="emb_mean_std_predict":
				emb_mean_std.Validate(model,test_emb,test_f0,test_len,"./model_prediction")
			elif mode=="emb_att_predict":
				emb_att.Validate(model,test_emb,test_f0,test_len,"./model_prediction")
			exit()
		model = None
		train_func = None
		if mode=="train_emb_lstm":
			model = embedding_lstm.EMB_LSTM(config.emb_size,config.voc_size,
				config.lstm_hidden_size,config.f0_dim,config.linear_h1)
			train_func = embedding_lstm.Train
		elif mode=="train_emb_mean_std":
			model = emb_mean_std.EMB_MEAN_STD(config.emb_size,config.voc_size,
			config.lstm_hidden_size,config.f0_dim,config.linear_shape,config.linear_mean,config.linear_std)
			train_func = emb_mean_std.Train
		elif mode=="train_emb_att":
			model = emb_att.EMB_ATT(config.emb_size,config.voc_size,
				config.lstm_hidden_size,config.f0_dim,config.linear_h1)
			train_func = emb_att.Train

		learning_rate = config.learning_rate
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		decay_step = config.decay_step
		decay_rate = config.decay_rate
		epoch_num = config.epoch_num
		train_func(
			train_emb,
			train_f0,
			train_len,
			test_emb,
			test_f0,
			test_len,
			model,
			optimizer,
			learning_rate,
			decay_step,
			decay_rate,
			epoch_num)

	elif mode=="train_feature_lstm" or mode=="feature_lstm_predict":
		desc_file = args.desc_file
		train_data = args.train_data
		train_label = args.train_label
		train_map = args.train_map
		test_data = args.test_data
		test_label = args.test_label
		test_map = args.test_map
		txt_file = args.txt_file

		# encode_feature = EncodeFeature(desc_file)
		# convert_feature(train_data,train_label,encode_feature,"./train_data_f0")
		# convert_feature(test_data,test_label,encode_feature,"./test_data_f0")

		os.system("mkdir lstm_data")
		print("--->collect data according to the data name")
		# word_index = word2index(txt_file,config.voc_size)
		# collect_utt_data("./train_data_f0",train_map,"./lstm_data/train",txt_file,word_index)
		# collect_utt_data("./test_data_f0",test_map,"./lstm_data/test",txt_file,word_index)
		print("--->get the numpy data for training")
		train_f0,train_feat,train_len = get_f0_feature("./lstm_data/train")
		test_f0,test_feat,test_len = get_f0_feature("./lstm_data/test")

		batch_num = train_f0.shape[0]/config.batch_size
		max_length = train_feat.shape[1]
		feat_num = train_feat.shape[2]

		train_f0 = train_f0[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_feat = train_feat[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,feat_num))
		train_len = train_len[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size))
		train_feat = torch.FloatTensor(train_feat.tolist())
		train_f0 = torch.FloatTensor(train_f0.tolist())
		train_len = torch.LongTensor(train_len.tolist())

		test_f0 = test_f0.reshape((len(test_f0),-1))
		test_feat = test_feat.reshape((len(test_feat),-1,feat_num))
		test_len = test_len
		# print(np.sum(test_len))
		test_feat = torch.FloatTensor(test_feat.tolist())
		test_f0 = torch.FloatTensor(test_f0.tolist())
		test_len = torch.LongTensor(test_len.tolist())

		if mode=="feature_lstm_predict":
			print("predicting...")
			model = torch.load("./my_best_model_.model")
			feature_lstm.Validate(model,test_feat,test_f0,test_len,"./model_prediction")
			exit()

		model = feature_lstm.FEAT_LSTM(feat_num,config.voc_size,
			config.lstm_hidden_size,config.f0_dim,config.linear_h1)
		learning_rate = config.learning_rate
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		decay_step = config.decay_step
		decay_rate = config.decay_rate
		epoch_num = config.epoch_num
		feature_lstm.Train(
			train_feat,
			train_f0,
			train_len,
			test_feat,
			test_f0,
			test_len,
			model,
			optimizer,
			learning_rate,
			decay_step,
			decay_rate,
			epoch_num)




