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
from utils import config
from utils import predict_mean_tool
import utils.data_processing
from utils.data_processing import EncodeFeature
from utils.data_processing import convert_feature
from utils.data_processing import concat_seq_feature
from utils.data_processing import word2index
from utils.data_processing import utt_content
from utils.data_processing import collect_utt_data
from utils.data_processing import create_train_val_data
from utils.data_processing import get_f0_embedding
from utils.data_processing import get_f0_feature
from utils.data_processing import get_f0_feature_list
from utils.data_processing import get_shape_mean_std
from utils.data_processing import parse_txt_file
from utils.data_processing import append_pos_to_feature
from utils.data_processing import one_hot_to_index
from utils.data_processing import get_syl_dic
from utils.data_processing import append_syl_to_feature
from utils.data_processing import get_pos_dic
from utils.data_processing import get_f0_dct
from utils.data_processing import normalize
from utils.data_processing import append_phrase_to_feature
from utils.data_processing import get_word_mean
from utils.data_processing import pos_refine
from model.mlp import MLP
from model import embedding_lstm
from model import feature_lstm
from model import emb_mean_std
from model import emb_att
from model import classify_mean
from model import emb_feat_lstm
from model import emb_pos_feat_lstm
from model import tone_lstm
from model import dct_lstm
from model import phrase_lstm

cuda_flag = config.cuda_flag
# from model import expand_emb
###########################################################
#GPU OPTION
###########################################################
if cuda_flag:
	import torch.backends.cudnn as cudnn
###########################################################
utils.data_processing.cuda_flag = cuda_flag
emb_pos_feat_lstm.cuda_flag = cuda_flag
tone_lstm.cuda_flag = cuda_flag
phrase_lstm.cuda_flag = cuda_flag

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
	parser.add_argument('--predict_file', dest='predict_file')
	parser.add_argument('--phrase_syl_dir', dest='phrase_syl_dir')
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
			" --desc_file ./feature_desc"+
			" --txt_file ../../txt.done.data"+
			" --train_data ./data/train/dct_0"+
			" --train_label ./data/train_data_f0_vector"+
			" --train_map ./data/train_syllable_map"+
			" --test_data ./data/dev/dct_0"+
			" --test_label ./data/dev_data_f0_vector"+
			" --test_map ./data/dev_syllable_map"+
			" --mode train_lstm")
		print("python train.py"+
			" --desc_file ../mandarine/gen_f0/train_dev_data_vector/feature_desc_vector"+
			" --txt_file ./data/txt.done.data-all"+
			" --train_data ../mandarine/gen_f0/train_dev_data_vector/train_data/dct_0"+
			" --train_label ../mandarine/gen_f0/train_dev_data_vector/train_data_f0_vector"+
			" --train_map ../mandarine/gen_f0/train_dev_data_vector/train_data/syllable_map"+
			" --test_data ../mandarine/gen_f0/train_dev_data_vector/dev_data/dct_0"+
			" --test_label ../mandarine/gen_f0/train_dev_data_vector/dev_data_f0_vector"+
			" --test_map ../mandarine/gen_f0/train_dev_data_vector/dev_data/syllable_map"+
			" --mode train_emb_lstm/emb_lstm_predict")
		print("python train.py"+
			" --desc_file ../mandarine/gen_f0/train_dev_data_vector/feature_desc_vector"+
			" --txt_file ./data/txt.done.data-all"+
			" --train_data ../mandarine/gen_f0/train_dev_data_vector/train_data/dct_0"+
			" --train_label ../mandarine/gen_f0/train_dev_data_vector/train_data_f0_vector"+
			" --train_map ../mandarine/gen_f0/train_dev_data_vector/train_data/syllable_map"+
			" --test_data ../mandarine/gen_f0/train_dev_data_vector/dev_data/dct_0"+
			" --test_label ../mandarine/gen_f0/train_dev_data_vector/dev_data_f0_vector"+
			" --test_map ../mandarine/gen_f0/train_dev_data_vector/dev_data/syllable_map"+
			" --phrase_syl_dir ../mandarine/gen_f0/phrase_dir/phrase_syllable"+
			" --mode train_phrase_lstm/emb_phrase_predict")
		
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
	elif "train_emb" in mode or ("emb" in mode and "feat" not in mode and "predict" in mode):
		desc_file = args.desc_file
		train_data = args.train_data
		train_label = args.train_label
		train_map = args.train_map
		test_data = args.test_data
		test_label = args.test_label
		test_map = args.test_map
		txt_file = args.txt_file

		normalize = True
		predict_val = "unnorm"

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
			elif predict_val=="unnorm":
				tmp_shape = np.loadtxt("./predict_shape",delimiter=" ")
				tmp_mean = np.loadtxt("./predict_mean",delimiter=" ").reshape((-1,1))
				tmp_std = np.loadtxt("./predict_std",delimiter=" ").reshape((-1,1))
				unnorm = tmp_shape*tmp_std+tmp_mean
				true_f0 = np.zeros(unnorm.shape)
				count = 0
				for utt in range(len(test_len)):
					true_f0[count:count+test_len[utt],:] = test_f0[utt,0:test_len[utt],:]
					count += test_len[utt]
				print("rmse: "+str(np.sqrt(np.square(unnorm-true_f0).mean(axis=1)).mean()))
				exit()

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

	elif mode=="train_feat_emb_lstm" or mode=="feat_emb_lstm_predict":
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
		train_emb = train_feat[:,:,-1].astype(np.int32)
		train_feat = train_feat[:,:,0:-1]
		test_emb = test_feat[:,:,-1].astype(np.int32)
		test_feat = test_feat[:,:,0:-1]

		batch_num = int(train_f0.shape[0]/config.batch_size)
		max_length = train_feat.shape[1]
		feat_num = train_feat.shape[2]

		train_f0 = train_f0[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_emb = train_emb[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_feat = train_feat[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,feat_num))
		train_len = train_len[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size))

		train_emb = torch.LongTensor(train_emb.tolist())
		train_feat = torch.FloatTensor(train_feat.tolist())
		train_f0 = torch.FloatTensor(train_f0.tolist())
		train_len = torch.LongTensor(train_len.tolist())

		test_f0 = test_f0.reshape((len(test_f0),-1))
		test_emb = test_emb.reshape((len(test_emb),-1))
		test_feat = test_feat.reshape((len(test_feat),-1,feat_num))
		test_len = test_len
		# print(np.sum(test_len))
		test_emb = torch.LongTensor(test_emb.tolist())
		test_feat = torch.FloatTensor(test_feat.tolist())
		test_f0 = torch.FloatTensor(test_f0.tolist())
		test_len = torch.LongTensor(test_len.tolist())

		if mode=="feature_lstm_predict":
			print("predicting...")
			model = torch.load("./my_best_model_.model")
			feature_lstm.Validate(model,test_feat,test_f0,test_len,"./model_prediction")
			exit()

		model = emb_feat_lstm.EMB_FEAT_LSTM(config.emb_size,feat_num,config.voc_size,
			config.lstm_hidden_size,config.f0_dim,config.linear_h1)
		learning_rate = config.learning_rate
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		decay_step = config.decay_step
		decay_rate = config.decay_rate
		epoch_num = config.epoch_num
		emb_feat_lstm.Train(
			train_emb,
			train_feat,
			train_f0,
			train_len,
			test_emb,
			test_feat,
			test_f0,
			test_len,
			model,
			optimizer,
			learning_rate,
			decay_step,
			decay_rate,
			epoch_num)

	elif mode=="train_feat_emb_pos_lstm" or mode=="feat_emb_pos_lstm_predict":
		desc_file = args.desc_file
		train_data = args.train_data
		train_label = args.train_label
		train_map = args.train_map
		test_data = args.test_data
		test_label = args.test_label
		test_map = args.test_map
		txt_file = args.txt_file
		pos_num = -1

		if config.update_data:
			############################################
			encode_feature = EncodeFeature(desc_file)

			# for tup in encode_feature.feature_pos:
			# 	print(tup[0] ),
			# 	print(str(tup[1][0])+" "+str(tup[1][1]))
			# exit()

			convert_feature(train_data,train_label,encode_feature,"./train_data_f0")
			convert_feature(test_data,test_label,encode_feature,"./test_data_f0")
			############################################

			############################################
			print("--->collect data according to the data name")
			os.system("mkdir lstm_data")
			word_index = word2index(txt_file,config.voc_size)
			collect_utt_data("./train_data_f0",train_map,"./lstm_data/train",txt_file,word_index)
			collect_utt_data("./test_data_f0",test_map,"./lstm_data/test",txt_file,word_index)
			############################################

			# parse_txt_file(txt_file,"./lstm_data/txt_token_pos")

			
			############################################
			pos_dic = get_pos_dic("./lstm_data/txt_token_pos")
			pos_num = len(pos_dic)+1
			append_pos_to_feature("./lstm_data/train","./lstm_data/txt_token_pos",pos_dic)
			append_pos_to_feature("./lstm_data/test","./lstm_data/txt_token_pos",pos_dic)
			############################################
		pos_num = 32

		print("--->get the numpy data for training")
		train_f0,train_feat,train_len = get_f0_feature("./lstm_data/train")
		test_f0,test_feat,test_len = get_f0_feature("./lstm_data/test")
		# shit_arr = []
		# for i in range(len(test_len)):
		# 	shit_arr.append(test_f0[i,0:test_len[i],:])
		# shit_arr = np.vstack(shit_arr)
		# np.savetxt("shit_arr",shit_arr,delimiter=" ",fmt="%.3f")
		# true_f0 = np.loadtxt("./dev_data_f0_vector_phrase",delimiter=" ")
		# print(np.sqrt(np.square(shit_arr-true_f0).mean(axis=1)).mean())
		# exit()

		if config.dct_flag:
			train_f0,train_mean,train_std = get_f0_dct(train_f0,train_len,config.dct_num,noramlize_flag=True)
			np.savetxt("tmp_mean_std",np.vstack((train_mean,train_std)))
			test_f0,_,_ = get_f0_dct(test_f0,test_len,config.dct_num,noramlize_flag=True)

		train_emb = train_feat[:,:,-2].astype(np.int32)
		train_pos = train_feat[:,:,-1].astype(np.int32)
		train_feat = train_feat[:,:,0:-2]

		test_emb = test_feat[:,:,-2].astype(np.int32)
		test_pos = test_feat[:,:,-1].astype(np.int32)
		test_feat = test_feat[:,:,0:-2]

		batch_num = int(train_f0.shape[0]/config.batch_size)
		max_length = train_feat.shape[1]
		feat_num = train_feat.shape[2]

		ori_train_f0 = train_f0
		ori_train_emb = train_emb
		ori_train_pos = train_pos
		ori_train_feat = train_feat
		ori_train_len = train_len

		train_f0 = train_f0[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_emb = train_emb[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_pos = train_pos[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_feat = train_feat[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,feat_num))
		train_len = train_len[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size))

		train_emb = torch.LongTensor(train_emb.tolist())
		train_pos = torch.LongTensor(train_pos.tolist())
		train_feat = torch.FloatTensor(train_feat.tolist())
		train_f0 = torch.FloatTensor(train_f0.tolist())
		train_len = torch.LongTensor(train_len.tolist())

		test_emb = test_emb.reshape((len(test_emb),-1))
		test_pos = test_pos.reshape((len(test_pos),-1))
		test_feat = test_feat.reshape((len(test_feat),-1,feat_num))
		test_f0 = test_f0.reshape((len(test_f0),-1))
		test_len = test_len
		# print(np.sum(test_len))
		test_emb = torch.LongTensor(test_emb.tolist())
		test_pos = torch.LongTensor(test_pos.tolist())
		test_feat = torch.FloatTensor(test_feat.tolist())
		test_f0 = torch.FloatTensor(test_f0.tolist())
		test_len = torch.LongTensor(test_len.tolist())

		if "predict" in mode:
			print("predicting...")
			# model = torch.load("./my_best_model.model")
			model = torch.load('./my_best_model.model', map_location=lambda storage, loc: storage)

			#############################################################
			# test_emb = torch.LongTensor(ori_train_emb.reshape((len(ori_train_emb),-1)).tolist())
			# test_pos = torch.LongTensor(ori_train_pos.reshape((len(ori_train_pos),-1)).tolist())
			# test_feat = torch.FloatTensor(ori_train_feat.reshape((len(ori_train_feat),max_length,feat_num)).tolist())
			# test_len = torch.LongTensor(ori_train_len.tolist())
			# test_f0 = torch.FloatTensor(ori_train_f0.reshape((len(ori_train_f0),-1)).tolist())
			#############################################################
			if config.dct_flag:
				dct_lstm.Validate(model,test_emb,test_pos,test_feat,test_f0,test_len,"./dct_emb_pos_feat_prediction")
			else:
				emb_pos_feat_lstm.Validate(model,test_emb,test_pos,test_feat,test_f0,test_len,"./emb_pos_feat_prediction")
			exit()
		if config.dct_flag:
			model = dct_lstm.DCT_LSTM(config.emb_size,config.pos_emb_size,feat_num,config.voc_size,pos_num,
				config.lstm_hidden_size,config.dct_num,config.linear_h1)
		else:
			model = emb_pos_feat_lstm.EMB_POS_FEAT_LSTM(config.emb_size,config.pos_emb_size,feat_num,config.voc_size,pos_num,
				config.lstm_hidden_size,config.f0_dim,config.linear_h1)
		##__init__(self,emb_size,pos_emb_size,feat_size,voc_size,pos_num,lstm_hidden_size,f0_dim,linear_h1)
		learning_rate = config.learning_rate
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		decay_step = config.decay_step
		decay_rate = config.decay_rate
		epoch_num = config.epoch_num
		if config.dct_flag:
			dct_lstm.Train(
			train_emb,
			train_pos,
			train_feat,
			train_f0,
			train_len,
			test_emb,
			test_pos,
			test_feat,
			test_f0,
			test_len,
			model,
			optimizer,
			learning_rate,
			decay_step,
			decay_rate,
			epoch_num)
		else:
			emb_pos_feat_lstm.Train(
				train_emb,
				train_pos,
				train_feat,
				train_f0,
				train_len,
				test_emb,
				test_pos,
				test_feat,
				test_f0,
				test_len,
				model,
				optimizer,
				learning_rate,
				decay_step,
				decay_rate,
				epoch_num)

	elif mode=="train_tone_lstm" or mode=="tone_lstm_predict":
		desc_file = args.desc_file
		train_data = args.train_data
		train_label = args.train_label
		train_map = args.train_map
		test_data = args.test_data
		test_label = args.test_label
		test_map = args.test_map
		txt_file = args.txt_file
		pos_num = -1

		############################################
		# encode_feature = EncodeFeature(desc_file)
		# convert_feature(train_data,train_label,encode_feature,"./train_data_f0")
		# convert_feature(test_data,test_label,encode_feature,"./test_data_f0")
		############################################

		############################################
		# print("--->collect data according to the data name")
		# os.system("mkdir lstm_data")
		# word_index = word2index(txt_file,config.voc_size)
		# collect_utt_data("./train_data_f0",train_map,"./lstm_data/train",txt_file,word_index)
		# collect_utt_data("./test_data_f0",test_map,"./lstm_data/test",txt_file,word_index)
		############################################

		# parse_txt_file(txt_file,"./lstm_data/txt_token_pos")

		
		############################################
		# pos_dic = get_pos_dic("./lstm_data/txt_token_pos")
		# pos_num = len(pos_dic)+1
		# append_pos_to_feature("./lstm_data/train","./lstm_data/txt_token_pos",pos_dic)
		# append_pos_to_feature("./lstm_data/test","./lstm_data/txt_token_pos",pos_dic)
		############################################
		pos_num = 32
		############################################


		############################################
		# consonant_dic,vowel_dic = get_syl_dic()
		# append_syl_to_feature("./lstm_data/train",train_map,consonant_dic,vowel_dic)
		# append_syl_to_feature("./lstm_data/test",test_map,consonant_dic,vowel_dic)
		# cons_num = len(consonant_dic)+1
		# vowel_num = len(vowel_dic)+1
		############################################
		cons_num = 24
		vowel_num = 38
		############################################

		print("--->get the numpy data for training")
		train_f0,train_feat,train_len = get_f0_feature("./lstm_data/train")
		test_f0,test_feat,test_len = get_f0_feature("./lstm_data/test")

		train_emb = train_feat[:,:,-4].astype(np.int32)
		train_pos = train_feat[:,:,-3].astype(np.int32)
		train_cons = train_feat[:,:,-2].astype(np.int32)
		train_vowel = train_feat[:,:,-1].astype(np.int32)
		train_feat = train_feat[:,:,0:-4]
		tmp_shape = train_feat.shape
		train_tone = one_hot_to_index(train_feat[:,:,3:8].astype(np.int32).reshape((-1,5))).reshape((tmp_shape[0],tmp_shape[1]))
		train_pretone = one_hot_to_index(train_feat[:,:,8:14].astype(np.int32).reshape((-1,6))).reshape((tmp_shape[0],tmp_shape[1]))
		train_postone = one_hot_to_index(train_feat[:,:,14:20].astype(np.int32).reshape((-1,6))).reshape((tmp_shape[0],tmp_shape[1]))
		# train_feat = np.delete(train_feat,range(3,20),2)

		test_emb = test_feat[:,:,-4].astype(np.int32)
		test_pos = test_feat[:,:,-3].astype(np.int32)
		test_cons = test_feat[:,:,-2].astype(np.int32)
		test_vowel = test_feat[:,:,-1].astype(np.int32)
		test_feat = test_feat[:,:,0:-4]
		tmp_shape = test_feat.shape
		test_tone = one_hot_to_index(test_feat[:,:,3:8].astype(np.int32).reshape((-1,5))).reshape((tmp_shape[0],tmp_shape[1]))
		test_pretone = one_hot_to_index(test_feat[:,:,8:14].astype(np.int32).reshape((-1,6))).reshape((tmp_shape[0],tmp_shape[1]))
		test_postone = one_hot_to_index(test_feat[:,:,14:20].astype(np.int32).reshape((-1,6))).reshape((tmp_shape[0],tmp_shape[1]))
		# test_feat = np.delete(test_feat,range(3,20),2)

		batch_num = int(train_f0.shape[0]/config.batch_size)
		max_length = train_emb.shape[1]
		feat_num = train_feat.shape[2]
		tone_num = 6
		pretone_num = 7
		postone_num = 7

		ori_train_f0 = train_f0
		ori_train_emb = train_emb
		ori_train_pos = train_pos
		ori_train_feat = train_feat
		ori_train_len = train_len

		train_f0 = train_f0[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_emb = train_emb[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_pos = train_pos[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_cons = train_cons[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_vowel = train_vowel[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_tone = train_tone[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_pretone = train_pretone[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_postone = train_postone[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_feat = train_feat[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,feat_num))
		train_len = train_len[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size))
		

		train_emb = torch.LongTensor(train_emb.tolist())
		train_pos = torch.LongTensor(train_pos.tolist())
		train_cons = torch.LongTensor(train_cons.tolist())
		train_vowel = torch.LongTensor(train_vowel.tolist())
		train_tone = torch.LongTensor(train_tone.tolist())
		train_pretone = torch.LongTensor(train_pretone.tolist())
		train_postone = torch.LongTensor(train_postone.tolist())
		train_feat = torch.FloatTensor(train_feat.tolist())
		train_f0 = torch.FloatTensor(train_f0.tolist())
		train_len = torch.LongTensor(train_len.tolist())

		test_emb = test_emb.reshape((len(test_emb),-1))
		test_pos = test_pos.reshape((len(test_pos),-1))
		test_cons = test_cons.reshape((len(test_cons),-1))
		test_vowel = test_vowel.reshape((len(test_vowel),-1))
		test_tone = test_tone.reshape((len(test_tone),-1))
		test_pretone = test_pretone.reshape((len(test_pretone),-1))
		test_postone = test_postone.reshape((len(test_postone),-1))
		test_feat = test_feat.reshape((len(test_feat),-1,feat_num))
		test_f0 = test_f0.reshape((len(test_f0),-1))
		test_len = test_len
		# print(np.sum(test_len))
		test_emb = torch.LongTensor(test_emb.tolist())
		test_pos = torch.LongTensor(test_pos.tolist())
		test_cons = torch.LongTensor(test_cons.tolist())
		test_vowel = torch.LongTensor(test_vowel.tolist())
		test_tone = torch.LongTensor(test_tone.tolist())
		test_pretone = torch.LongTensor(test_pretone.tolist())
		test_postone = torch.LongTensor(test_postone.tolist())
		test_feat = torch.FloatTensor(test_feat.tolist())
		test_f0 = torch.FloatTensor(test_f0.tolist())
		test_len = torch.LongTensor(test_len.tolist())

		if "predict" in mode:
			print("predicting...")
			model = torch.load("my_best_model_.model")

			#############################################################
			# test_emb = torch.LongTensor(ori_train_emb.reshape((len(ori_train_emb),-1)).tolist())
			# test_pos = torch.LongTensor(ori_train_pos.reshape((len(ori_train_pos),-1)).tolist())
			# test_feat = torch.FloatTensor(ori_train_feat.reshape((len(ori_train_feat),max_length,feat_num)).tolist())
			# test_len = torch.LongTensor(ori_train_len.tolist())
			# test_f0 = torch.FloatTensor(ori_train_f0.reshape((len(ori_train_f0),-1)).tolist())
			#############################################################

			# tone_lstm.Validate(model,test_emb,test_pos,test_pretone,test_tone,test_postone,test_feat,test_f0,test_len,"./emb_pos_feat_prediction")
			tone_lstm.Validate(model,test_emb,test_pos,test_cons,test_vowel,test_pretone,test_tone,test_postone,
				test_feat,test_f0,test_len,"./tone_lstm_prediction")
			exit()
		model = tone_lstm.TONE_LSTM(config.emb_size,config.pos_emb_size,config.tone_emb_size,
			cons_num,vowel_num,pretone_num,tone_num,postone_num,feat_num,config.voc_size,pos_num,
			config.lstm_hidden_size,config.f0_dim,config.linear_h1)
		##__init__(self,emb_size,pos_emb_size,tone_emb_size,pretone_num,tone_num,postone_num,feat_size,voc_size,pos_num,lstm_hidden_size,f0_dim,linear_h1)
		learning_rate = config.learning_rate
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		decay_step = config.decay_step
		decay_rate = config.decay_rate
		epoch_num = config.epoch_num
		tone_lstm.Train(
			train_emb,
			train_pos,
			train_cons,
			train_vowel,
			train_pretone,
			train_tone,
			train_postone,
			train_feat,
			train_f0,
			train_len,
			test_emb,
			test_pos,
			test_cons,
			test_vowel,
			test_pretone,
			test_tone,
			test_postone,
			test_feat,
			test_f0,
			test_len,
			model,
			optimizer,
			learning_rate,
			decay_step,
			decay_rate,
			epoch_num)

	elif mode=="train_phrase_lstm" or mode=="phrase_lstm_predict":
		desc_file = args.desc_file
		train_data = args.train_data
		train_label = args.train_label
		train_map = args.train_map
		test_data = args.test_data
		test_label = args.test_label
		test_map = args.test_map
		txt_file = args.txt_file
		phrase_syl_dir = args.phrase_syl_dir
		pos_num = -1
		if config.update_data:
			############################################
			encode_feature = EncodeFeature(desc_file)
			convert_feature(train_data,train_label,encode_feature,"./train_data_f0")
			convert_feature(test_data,test_label,encode_feature,"./test_data_f0")
			############################################

			############################################
			print("--->collect data according to the data name")
			os.system("mkdir lstm_data")
			word_index = word2index(txt_file,config.voc_size)
			collect_utt_data("./train_data_f0",train_map,"./lstm_data/train",txt_file,word_index)
			collect_utt_data("./test_data_f0",test_map,"./lstm_data/test",txt_file,word_index)
			############################################

			# parse_txt_file(txt_file,"./lstm_data/txt_token_pos")

			
			############################################
			pos_refine("./lstm_data/pos_convert_map","./lstm_data/txt_token_pos","./lstm_data/refine_txt_token_pos")
			pos_file = "./lstm_data/refine_txt_token_pos"
			pos_dic = get_pos_dic(pos_file)
			pos_num = len(pos_dic)+1
			print("pos number: "+str(pos_num))
			append_pos_to_feature("./lstm_data/train",pos_file,pos_dic)
			append_pos_to_feature("./lstm_data/test",pos_file,pos_dic)
			############################################
			# pos_num = 32
			############################################


			############################################
			consonant_dic,vowel_dic = get_syl_dic()
			append_syl_to_feature("./lstm_data/train",train_map,consonant_dic,vowel_dic)
			append_syl_to_feature("./lstm_data/test",test_map,consonant_dic,vowel_dic)
			cons_num = len(consonant_dic)+1
			vowel_num = len(vowel_dic)+1
			############################################
			# cons_num = 24
			# vowel_num = 38
			############################################

			############################################
			append_phrase_to_feature("./lstm_data/train",phrase_syl_dir)
			append_phrase_to_feature("./lstm_data/test",phrase_syl_dir)
			exit()
			############################################

		pos_num = 18
		cons_num = 24
		vowel_num = 38

		print("--->get the numpy data for training")
		train_f0,train_feat,train_len = get_f0_feature("./lstm_data/train")
		test_f0,test_feat,test_len = get_f0_feature("./lstm_data/test")

		############################################
		#append the previous f0 and next f0 value
		# train_pre_f0 = np.copy(train_f0[:,:,-1].reshape((train_f0.shape[0],train_f0.shape[1],1)))
		# train_pre_f0[:,1:,:] = train_pre_f0[:,0:-1,:]
		# train_pre_f0[:,0,0] = 0
		# train_post_f0 = np.copy(train_f0[:,:,0].reshape((train_f0.shape[0],train_f0.shape[1],1)))
		# train_post_f0[:,0:-1,:] = train_post_f0[:,1:,:]
		# train_post_f0[:,-1,0] = 0
		# train_f0 = np.concatenate((train_pre_f0,train_f0,train_post_f0),axis=2)

		# test_pre_f0 = np.copy(test_f0[:,:,-1].reshape((test_f0.shape[0],test_f0.shape[1],1)))
		# test_pre_f0[:,1:,:] = test_pre_f0[:,0:-1,:]
		# test_pre_f0[:,0,0] = 0
		# test_post_f0 = np.copy(test_f0[:,:,0].reshape((test_f0.shape[0],test_f0.shape[1],1)))
		# test_post_f0[:,0:-1,:] = test_post_f0[:,1:,:]
		# test_post_f0[:,-1,0] = 0
		# test_f0 = np.concatenate((test_pre_f0,test_f0,test_post_f0),axis=2)

		# print(train_f0[0,0,:])
		# exit()
		############################################
		# if predict mean
		train_f0 = train_f0.mean(axis=2).reshape((train_f0.shape[0],train_f0.shape[1],1))
		test_f0 = test_f0.mean(axis=2).reshape((test_f0.shape[0],test_f0.shape[1],1))
		############################################
		# get the mean f0 for word
		# train_emb = train_feat[:,:,-10].astype(np.int32)
		# train_word_mean,word_mean_dic = get_word_mean(train_emb.flatten(),train_f0.reshape((-1,10)),config.voc_size)
		# train_word_mean = train_word_mean.reshape(train_f0.shape)
		# train_f0 = (train_f0.mean(axis=2)-train_word_mean.mean(axis=2)).reshape((train_f0.shape[0],train_f0.shape[1],1))

		# test_emb = test_feat[:,:,-10].astype(np.int32)
		# test_word_mean,_ = get_word_mean(test_emb.flatten(),test_f0.reshape((-1,10)),config.voc_size,word_mean_dic)
		# test_word_mean = test_word_mean.reshape(test_f0.shape)
		# test_f0 = (test_f0.mean(axis=2)-test_word_mean.mean(axis=2)).reshape((test_f0.shape[0],test_f0.shape[1],1))
		############################################



		train_emb = train_feat[:,:,-14].astype(np.int32)
		train_pos = train_feat[:,:,-13:-8].astype(np.int32)
		train_pos_feat = train_pos[:,:,3:]
		train_pos = train_pos[:,:,0:3]
		train_cons = train_feat[:,:,-8].astype(np.int32)
		train_vowel = train_feat[:,:,-7].astype(np.int32)
		train_phrase = train_feat[:,:,-6:]
		train_feat = train_feat[:,:,0:-14]
		tmp_shape = train_feat.shape
		train_tone = one_hot_to_index(train_feat[:,:,3:8].astype(np.int32).reshape((-1,5))).reshape((tmp_shape[0],tmp_shape[1]))
		train_pretone = one_hot_to_index(train_feat[:,:,8:14].astype(np.int32).reshape((-1,6))).reshape((tmp_shape[0],tmp_shape[1]))
		train_postone = one_hot_to_index(train_feat[:,:,14:20].astype(np.int32).reshape((-1,6))).reshape((tmp_shape[0],tmp_shape[1]))
		##delete pitch feature
		# train_feat = np.delete(train_feat,range(35,39),2)

		# print(train_pos_feat.shape)
		# exit()

		test_emb = test_feat[:,:,-14].astype(np.int32)
		test_pos = test_feat[:,:,-13:-8].astype(np.int32)
		test_pos_feat = test_pos[:,:,3:]
		test_pos = test_pos[:,:,0:3]
		test_cons = test_feat[:,:,-8].astype(np.int32)
		test_vowel = test_feat[:,:,-7].astype(np.int32)
		test_phrase = test_feat[:,:,-6:]
		test_feat = test_feat[:,:,0:-14]
		tmp_shape = test_feat.shape
		test_tone = one_hot_to_index(test_feat[:,:,3:8].astype(np.int32).reshape((-1,5))).reshape((tmp_shape[0],tmp_shape[1]))
		test_pretone = one_hot_to_index(test_feat[:,:,8:14].astype(np.int32).reshape((-1,6))).reshape((tmp_shape[0],tmp_shape[1]))
		test_postone = one_hot_to_index(test_feat[:,:,14:20].astype(np.int32).reshape((-1,6))).reshape((tmp_shape[0],tmp_shape[1]))
		##delete pitch feature
		# test_feat = np.delete(test_feat,range(35,39),2)

		batch_num = int(train_f0.shape[0]/config.batch_size)
		max_length = train_emb.shape[1]
		feat_num = train_feat.shape[2]
		phrase_num = train_phrase.shape[2]
		tone_num = 6
		pretone_num = 7
		postone_num = 7
		pos_feat_num = 2

		# ori_train_f0 = train_f0
		# ori_train_emb = train_emb
		# ori_train_pos = train_pos
		# ori_train_feat = train_feat
		# ori_train_len = train_len

		train_f0 = train_f0[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_emb = train_emb[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_pos = train_pos[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,-1))
		train_pos_feat = train_pos_feat[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,-1))
		train_cons = train_cons[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_vowel = train_vowel[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_tone = train_tone[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_pretone = train_pretone[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_postone = train_postone[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_feat = train_feat[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,feat_num))
		train_len = train_len[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size))
		train_phrase = train_phrase[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,phrase_num))


		# print(train_pos_feat.shape)

		train_emb = torch.LongTensor(train_emb.tolist())
		train_pos = torch.LongTensor(train_pos.tolist())
		train_pos_feat = torch.FloatTensor(train_pos_feat.tolist())
		train_cons = torch.LongTensor(train_cons.tolist())
		train_vowel = torch.LongTensor(train_vowel.tolist())
		train_tone = torch.LongTensor(train_tone.tolist())
		train_pretone = torch.LongTensor(train_pretone.tolist())
		train_postone = torch.LongTensor(train_postone.tolist())
		train_feat = torch.FloatTensor(train_feat.tolist())
		train_f0 = torch.FloatTensor(train_f0.tolist())
		train_len = torch.LongTensor(train_len.tolist())
		train_phrase = torch.FloatTensor(train_phrase.tolist())

		test_emb = test_emb.reshape((len(test_emb),-1))
		test_pos = test_pos.reshape((test_pos.shape[0],test_pos.shape[1],-1))
		test_pos_feat = test_pos_feat.reshape((test_pos_feat.shape[0],test_pos_feat.shape[1],-1))
		test_cons = test_cons.reshape((len(test_cons),-1))
		test_vowel = test_vowel.reshape((len(test_vowel),-1))
		test_tone = test_tone.reshape((len(test_tone),-1))
		test_pretone = test_pretone.reshape((len(test_pretone),-1))
		test_postone = test_postone.reshape((len(test_postone),-1))
		test_feat = test_feat.reshape((len(test_feat),-1,feat_num))
		test_f0 = test_f0.reshape((len(test_f0),-1))
		test_len = test_len
		test_phrase = test_phrase.reshape((len(test_phrase),-1,phrase_num))

		# print(np.sum(test_len))
		test_emb = torch.LongTensor(test_emb.tolist())
		test_pos = torch.LongTensor(test_pos.tolist())
		test_pos_feat = torch.FloatTensor(test_pos_feat.tolist())
		test_cons = torch.LongTensor(test_cons.tolist())
		test_vowel = torch.LongTensor(test_vowel.tolist())
		test_tone = torch.LongTensor(test_tone.tolist())
		test_pretone = torch.LongTensor(test_pretone.tolist())
		test_postone = torch.LongTensor(test_postone.tolist())
		test_feat = torch.FloatTensor(test_feat.tolist())
		test_f0 = torch.FloatTensor(test_f0.tolist())
		test_len = torch.LongTensor(test_len.tolist())
		test_phrase = torch.FloatTensor(test_phrase.tolist())

		if "predict" in mode:
			print("predicting...")
			model = torch.load("my_best_model.model")
			# model = torch.load('./best_phrase_model', map_location=lambda storage, loc: storage)

			#############################################################
			# test_emb = torch.LongTensor(ori_train_emb.reshape((len(ori_train_emb),-1)).tolist())
			# test_pos = torch.LongTensor(ori_train_pos.reshape((len(ori_train_pos),-1)).tolist())
			# test_feat = torch.FloatTensor(ori_train_feat.reshape((len(ori_train_feat),max_length,feat_num)).tolist())
			# test_len = torch.LongTensor(ori_train_len.tolist())
			# test_f0 = torch.FloatTensor(ori_train_f0.reshape((len(ori_train_f0),-1)).tolist())
			#############################################################

			# tone_lstm.Validate(model,test_emb,test_pos,test_pretone,test_tone,test_postone,test_feat,test_f0,test_len,"./emb_pos_feat_prediction")
			phrase_lstm.Validate(model,test_emb,test_pos,test_pos_feat,test_cons,test_vowel,test_pretone,test_tone,test_postone,
				test_feat,test_phrase,test_f0,test_len,"./phrase_lstm_prediction")
			exit()
		#############################################################
		# model = phrase_lstm.PHRASE_LSTM(config.emb_size,config.pos_emb_size,config.tone_emb_size,
		# 	cons_num,vowel_num,pretone_num,tone_num,postone_num,feat_num,phrase_num,config.voc_size,pos_num,pos_feat_num,
		# 	config.lstm_hidden_size,config.f0_dim,config.linear_h1)
		#############################################################
		##if predict mean
		model = phrase_lstm.PHRASE_MEAN_LSTM(config.emb_size,config.pos_emb_size,config.tone_emb_size,
			cons_num,vowel_num,pretone_num,tone_num,postone_num,feat_num,phrase_num,config.voc_size,pos_num,pos_feat_num,
			config.lstm_hidden_size,config.f0_dim,config.linear_h1)
		#############################################################
		learning_rate = config.learning_rate
		param_list = []
		for name,param in model.named_parameters():
			if name in ["embed.weight","pos_embed","cons_embed","vowel_embed"]:
				# print("find embed.weight")
				param_list.append({'params': param, 'lr': 10*learning_rate,"my_name":name})
			else:
				param_list.append({'params': param,"my_name":name})

		# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		optimizer = optim.Adam(param_list, lr=learning_rate)
		# tmp_param = [name for name,param in model.named_parameters()]
		# print(tmp_param)
		decay_step = config.decay_step
		decay_rate = config.decay_rate
		epoch_num = config.epoch_num
		phrase_lstm.Train(
			train_emb,
			train_pos,
			train_pos_feat,
			train_cons,
			train_vowel,
			train_pretone,
			train_tone,
			train_postone,
			train_feat,
			train_phrase,
			train_f0,
			train_len,
			test_emb,
			test_pos,
			test_pos_feat,
			test_cons,
			test_vowel,
			test_pretone,
			test_tone,
			test_postone,
			test_feat,
			test_phrase,
			test_f0,
			test_len,
			model,
			optimizer,
			learning_rate,
			decay_step,
			decay_rate,
			epoch_num)
		


