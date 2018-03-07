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
from utils.data_processing import parse_txt_file_pos
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
from utils.data_processing import get_dep_dic
from utils.data_processing import append_dep_to_feature
from utils.data_processing import dep_refine
from utils.data_processing import generate_word_embedding
from utils.data_processing import new_word2index
from utils.data_processing import save_dic
from utils.data_processing import decompose_zh_syl
from utils.data_processing import read_dic

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
from model import seq2seq


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
seq2seq.cuda_flag = cuda_flag

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
	parser.add_argument('--phrase_syl_dir', dest='phrase_syl_dir')
	parser.add_argument('--model', dest='model',default="./my_best_model.model")
	parser.add_argument('--predict_file',dest='predict_file',default="./predict_f0")
	parser.add_argument('--out_dir',dest='out_dir')
	parser.add_argument('--timeline',dest='timeline',type=int,default=0)
	parser.add_argument('--voice_dir',dest='voice_dir')
	parser.add_argument('--data_dir',dest='data_dir',default="")
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
			" --desc_file ../mandarine/dnn_data_dir/feature_desc_vector"+
			" --txt_file ../mandarine/txt.done.data-all"+
			" --train_data ../mandarine/dnn_data_dir/train_test_data/train_data/train_feat"+
			" --train_label ../mandarine/dnn_data_dir/train_test_data/train_data/train_f0"+
			" --train_map ../mandarine/dnn_data_dir/train_test_data/train_data/train_syllable_map"+
			" --test_data ../mandarine/dnn_data_dir/train_test_data/test_data/test_feat"+
			" --test_label ../mandarine/dnn_data_dir/train_test_data/test_data/test_f0"+
			" --test_map ../mandarine/dnn_data_dir/train_test_data/test_data/test_syllable_map"+
			" --phrase_syl_dir ../mandarine/dnn_data_dir/phrase_dir/phrase_syllable"+
			" --mode phrase_lstm_predict"+
			" --model ./trained_model/syllable_lstm"+
			" --predict_file model_prediction/syllable_lstm_f0")
		print("python train.py"+
			" --mode phrase_lstm_predict"+
			" --data_dir ../mandarine/dnn_data_dir"+
			" --voice_dir ../mandarine/cmu_yue_wdy_cn"+
			" --txt_file ../mandarine/txt.done.data.word"+
			# " --desc_file ../cantonese/dnn_data_dir/feature_desc_vector"+
			# " --train_data ../cantonese/dnn_data_dir/train_test_data/train_data/train_feat"+
			# " --train_label ../cantonese/dnn_data_dir/train_test_data/train_data/train_f0"+
			# " --train_map ../cantonese/dnn_data_dir/train_test_data/train_data/train_syllable_map"+
			# " --test_data ../cantonese/dnn_data_dir/train_test_data/test_data/test_feat"+
			# " --test_label ../cantonese/dnn_data_dir/train_test_data/test_data/test_f0"+
			# " --test_map ../cantonese/dnn_data_dir/train_test_data/test_data/test_syllable_map"+
			# " --phrase_syl_dir ../cantonese/dnn_data_dir/phrase_dir/phrase_syllable"+
			" --model ./trained_model/add_lstm_model"+
			" --predict_file add_lstm_predict_f0")
			# " --timeline 0/1(optional 0 default)"+
			# " --out_dir(mandatory if timeline=1)")

	elif mode=="phrase_lstm_train" or mode=="phrase_lstm_predict":
		if args.data_dir != "":
			data_dir = args.data_dir
			voice_dir = args.voice_dir
			desc_file = data_dir+"/feature_desc_vector"
			train_data = data_dir+"/train_test_data/train_data/train_feat"
			train_label = data_dir+"/train_test_data/train_data/train_f0"
			train_map = data_dir+"/train_test_data/train_data/train_syllable_map"
			############################################
			test_data = data_dir+"/train_test_data/test_data/test_feat"
			test_label = data_dir+"/train_test_data/test_data/test_f0"
			test_map = data_dir+"/train_test_data/test_data/test_syllable_map"
			############################################
			# test_data = "../experiment/mandarine/dnn_data_dir/all_data/test_feat"
			# test_label = "../experiment/mandarine/dnn_data_dir/all_data/test_f0"
			# test_map = "../experiment/mandarine/dnn_data_dir/all_data/test_syllable_map"
			############################################
			phrase_syl_dir = data_dir+"/phrase_dir/phrase_syllable"
		else:
			desc_file = args.desc_file
			train_data = args.train_data
			train_label = args.train_label
			train_map = args.train_map
			############################################
			test_data = args.test_data
			test_label = args.test_label
			test_map = args.test_map
			phrase_syl_dir = args.phrase_syl_dir
		txt_file = args.txt_file
		pos_num = -1
		if config.update_data:
			os.system("mkdir dic_dir")
			############################################
			encode_feature = EncodeFeature(desc_file)
			convert_feature(train_data,train_label,encode_feature,"./train_data_f0")
			convert_feature(test_data,test_label,encode_feature,"./test_data_f0")
			############################################

			############################################
			print("--->collect data according to the data name")
			os.system("mkdir lstm_data")
			# word_index = word2index(txt_file,config.voc_size)
			word_index = new_word2index(txt_file,"./lstm_data/emb_dic")
			save_dic(word_index,"dic_dir/word_dic")

			config.voc_size = len(word_index)+1
			print("vocab size: "+str(config.voc_size))

			with open("./lstm_data/word_dic","w+") as f:
				for word,idx in word_index.items():
					f.write(word+" "+str(idx)+"\n")

			collect_utt_data("./train_data_f0",train_map,"./lstm_data/train",txt_file,word_index)
			############################################
			collect_utt_data("./test_data_f0",test_map,"./lstm_data/test",txt_file,word_index)
			############################################
			# collect_utt_data("./test_data_f0",test_map,"./lstm_data/test","../experiment/mandarine/txt.done.data.test_word",word_index)
			############################################

			generate_word_embedding("./lstm_data/word_dic","./lstm_data/emb_dic",config.voc_size,300,"./lstm_data/pretrain_emb")
			############################################

			# parse_txt_file_pos(txt_file,"./lstm_data/txt_token_pos")

			
			############################################
			pos_refine("./lstm_data/pos_convert_map","./lstm_data/txt_token_pos","./lstm_data/refine_txt_token_pos")
			pos_file = "./lstm_data/refine_txt_token_pos"
			pos_dic = get_pos_dic(pos_file)
			save_dic(pos_dic,"dic_dir/pos_dic")

			pos_num = len(pos_dic)+1
			print("pos number: "+str(pos_num))
			append_pos_to_feature("./lstm_data/train",pos_file,pos_dic)
			############################################
			append_pos_to_feature("./lstm_data/test",pos_file,pos_dic)
			############################################
			# append_pos_to_feature("./lstm_data/test","../experiment/mandarine/refine_pos",pos_dic)
			############################################
			# pos_num = 32
			############################################


			############################################
			consonant_dic,vowel_dic,vowel_char_dic = get_syl_dic()
			save_dic(consonant_dic,"dic_dir/cons_dic")
			save_dic(vowel_dic,"dic_dir/vowel_dic")
			save_dic(vowel_char_dic,"dic_dir/vowel_char_dic")

			append_syl_to_feature("./lstm_data/train",train_map,consonant_dic,vowel_dic,vowel_char_dic)
			append_syl_to_feature("./lstm_data/test",test_map,consonant_dic,vowel_dic,vowel_char_dic)
			print(vowel_char_dic)
			cons_num = len(consonant_dic)+1
			vowel_num = len(vowel_dic)+1
			vowel_ch_num = len(vowel_char_dic)+1
			############################################
			# cons_num = 24
			# vowel_num = 38
			############################################

			############################################
			append_phrase_to_feature("./lstm_data/train",phrase_syl_dir)
			############################################
			append_phrase_to_feature("./lstm_data/test",phrase_syl_dir)
			############################################
			# append_phrase_to_feature("./lstm_data/test","../experiment/mandarine/dnn_data_dir/phrase_dir/phrase_syllable")
			############################################

			############################################
			##dependency feature
			dep_refine("./lstm_data/dep_convert_map","./lstm_data/txt_token_dep","./lstm_data/refine_txt_token_dep")
			dep_file = "./lstm_data/refine_txt_token_dep"
			# parse_txt_file_dep(txt_file,dep_file)
			dep_dic = get_dep_dic(dep_file,"./lstm_data/dependency_dic")
			save_dic(dep_dic,"dic_dir/dep_dic")

			dep_num = len(dep_dic)*2
			print("dep_num: "+str(dep_num))
			append_dep_to_feature("./lstm_data/train",dep_file,dep_dic)
			############################################
			append_dep_to_feature("./lstm_data/test",dep_file,dep_dic)
			############################################
			# append_dep_to_feature("./lstm_data/test","../experiment/mandarine/refine_dep",dep_dic)
			############################################
			exit()
			############################################

		pos_num = 18
		cons_num = 24
		vowel_num = 38
		dep_num = 30*2
		config.voc_size = 3601
		vowel_ch_num = 10

		print("--->get the numpy data for training")
		train_f0,train_feat,train_len = get_f0_feature("./lstm_data/train")
		############################################
		test_f0,test_feat,test_len = get_f0_feature("./lstm_data/test")
		############################################
		# test_f0,test_feat,test_len = get_f0_feature("./self_generated_test_data/data_dir")
		############################################

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
		# train_std = train_f0.std(axis=2).reshape((train_f0.shape[0],train_f0.shape[1],1))
		# test_std = test_f0.std(axis=2).reshape((test_f0.shape[0],test_f0.shape[1],1))
		# train_mean = train_f0.mean(axis=2).reshape((train_f0.shape[0],train_f0.shape[1],1))
		# test_mean = test_f0.mean(axis=2).reshape((test_f0.shape[0],test_f0.shape[1],1))
		# train_f0 = (train_f0-train_mean)/(train_std+0.0000001)
		# test_f0 = (test_f0-test_mean)/(test_std+0.0000001)
		# train_f0 = train_mean
		# test_f0 = test_mean
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



		train_emb = train_feat[:,:,68].astype(np.int32)
		train_pos = train_feat[:,:,69:74].astype(np.int32)
		train_pos_feat = train_pos[:,:,3:]
		train_pos = train_pos[:,:,0:3]
		train_cons = train_feat[:,:,74].astype(np.int32)
		train_vowel = train_feat[:,:,75:80].astype(np.int32)
		train_phrase = train_feat[:,:,80:86]
		train_dep = train_feat[:,:,86:146]
		train_feat = train_feat[:,:,0:68]
		tmp_shape = train_feat.shape
		train_tone = one_hot_to_index(train_feat[:,:,3:8].astype(np.int32).reshape((-1,5))).reshape((tmp_shape[0],tmp_shape[1]))
		train_pretone = one_hot_to_index(train_feat[:,:,8:14].astype(np.int32).reshape((-1,6))).reshape((tmp_shape[0],tmp_shape[1]))
		train_postone = one_hot_to_index(train_feat[:,:,14:20].astype(np.int32).reshape((-1,6))).reshape((tmp_shape[0],tmp_shape[1]))
		##delete pitch feature
		train_feat = np.delete(train_feat,range(21,27),2)

		# print(train_pos_feat.shape)
		# exit()

		test_emb = test_feat[:,:,68].astype(np.int32)
		test_pos = test_feat[:,:,69:74].astype(np.int32)
		test_pos_feat = test_pos[:,:,3:]
		test_pos = test_pos[:,:,0:3]
		test_cons = test_feat[:,:,74].astype(np.int32)
		test_vowel = test_feat[:,:,75:80].astype(np.int32)
		test_phrase = test_feat[:,:,80:86]
		test_dep = test_feat[:,:,86:146]
		# print(test_feat.shape)
		test_feat = test_feat[:,:,0:68]
		tmp_shape = test_feat.shape
		test_tone = one_hot_to_index(test_feat[:,:,3:8].astype(np.int32).reshape((-1,5))).reshape((tmp_shape[0],tmp_shape[1]))
		test_pretone = one_hot_to_index(test_feat[:,:,8:14].astype(np.int32).reshape((-1,6))).reshape((tmp_shape[0],tmp_shape[1]))
		test_postone = one_hot_to_index(test_feat[:,:,14:20].astype(np.int32).reshape((-1,6))).reshape((tmp_shape[0],tmp_shape[1]))
		##delete pitch feature
		test_feat = np.delete(test_feat,range(21,27),2)

		batch_num = int(train_f0.shape[0]/config.batch_size)
		max_length = train_emb.shape[1]
		feat_num = train_feat.shape[2]
		phrase_num = train_phrase.shape[2]
		tone_num = 6
		pretone_num = 7
		postone_num = 7
		pos_feat_num = 2

		ori_train_f0 = train_f0
		ori_train_emb = train_emb
		ori_train_pos = train_pos
		ori_train_pos_feat = train_pos_feat
		ori_train_cons = train_cons
		ori_train_vowel = train_vowel
		ori_train_tone = train_tone
		ori_train_pretone = train_pretone
		ori_train_postone = train_postone
		ori_train_feat = train_feat
		ori_train_len = train_len
		ori_train_phrase = train_phrase
		ori_train_dep = train_dep

		train_f0 = train_f0[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,-1))
		train_emb = train_emb[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_pos = train_pos[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,-1))
		train_pos_feat = train_pos_feat[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,-1))
		train_cons = train_cons[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_vowel = train_vowel[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,-1))
		train_tone = train_tone[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_pretone = train_pretone[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_postone = train_postone[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,-1))
		train_feat = train_feat[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,feat_num))
		train_len = train_len[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size))
		train_phrase = train_phrase[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,phrase_num))
		train_dep = train_dep[0:batch_num*config.batch_size].reshape((batch_num,config.batch_size,max_length,dep_num))


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
		train_dep = torch.FloatTensor(train_dep.tolist())

		test_emb = test_emb.reshape((len(test_emb),-1))
		test_pos = test_pos.reshape((test_pos.shape[0],test_pos.shape[1],-1))
		test_pos_feat = test_pos_feat.reshape((test_pos_feat.shape[0],test_pos_feat.shape[1],-1))
		test_cons = test_cons.reshape((len(test_cons),-1))
		test_vowel = test_vowel.reshape((len(test_vowel),-1,5))
		test_tone = test_tone.reshape((len(test_tone),-1))
		test_pretone = test_pretone.reshape((len(test_pretone),-1))
		test_postone = test_postone.reshape((len(test_postone),-1))
		# print(test_feat.shape)
		test_feat = test_feat.reshape((len(test_feat),-1,feat_num))
		test_f0 = test_f0.reshape((len(test_f0),test_emb.shape[1],-1))
		test_len = test_len
		test_phrase = test_phrase.reshape((len(test_phrase),-1,phrase_num))
		# print(test_dep.shape)
		test_dep = test_dep.reshape((len(test_dep),-1,dep_num))

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
		test_dep = torch.FloatTensor(test_dep.tolist())

		if "predict" in mode:
			print("predicting...")
			trained_model = args.model
			predict_file = args.predict_file
			model = torch.load(trained_model, map_location=lambda storage, loc: storage)
			# for name,param in model.named_parameters():
			# 	print(name)

			#############################################################
			# test_emb = torch.LongTensor(ori_train_emb.reshape((len(ori_train_emb),-1)).tolist())
			# test_pos = torch.LongTensor(ori_train_pos.reshape((len(ori_train_pos),-1)).tolist())
			# test_pos_feat = torch.FloatTensor(ori_train_pos_feat.tolist())
			# test_cons = torch.LongTensor(ori_train_cons.tolist())
			# test_vowel = torch.LongTensor(ori_train_vowel.tolist())
			# test_tone = torch.LongTensor(ori_train_tone.tolist())
			# test_pretone = torch.LongTensor(ori_train_pretone.tolist())
			# test_postone = torch.LongTensor(ori_train_postone.tolist())
			# test_feat = torch.FloatTensor(ori_train_feat.reshape((len(ori_train_feat),max_length,feat_num)).tolist())
			# test_len = torch.LongTensor(ori_train_len.tolist())
			# test_f0 = torch.FloatTensor(ori_train_f0.reshape((len(ori_train_f0),-1)).tolist())
			# test_phrase = torch.FloatTensor(ori_train_phrase.tolist())
			# test_dep = torch.FloatTensor(ori_train_dep.tolist())
			#############################################################

			
			# tone_lstm.Validate(model,test_emb,test_pos,test_pretone,test_tone,test_postone,test_feat,test_f0,test_len,"./emb_pos_feat_prediction")
			loss,_ = phrase_lstm.Validate(model,test_emb,test_pos,test_pos_feat,test_cons,test_vowel,test_pretone,test_tone,test_postone,
				test_feat,test_phrase,test_dep,test_f0,test_len,predict_file)
			print("rmse: "+str(loss))

			# timeline = args.timeline
			# if timeline == 1:
			# 	out_dir = args.out_dir
			# 	os.system("python dnn_prediction_statistics.py"+
			# 		" --mode stat"+
			# 		" --voice_dir "+voice_dir+
			# 		" --data_dir "+data_dir+
			# 		" --predict_file "+predict_file+
			# 		" --out_dir "+out_dir)
			exit()
		#############################################################
		model = phrase_lstm.PHRASE_LSTM(config.emb_size,config.pos_emb_size,config.tone_emb_size,
			cons_num,vowel_num,vowel_ch_num,pretone_num,tone_num,postone_num,feat_num,phrase_num,dep_num,config.voc_size,pos_num,pos_feat_num,
			config.lstm_hidden_size,config.f0_dim,config.linear_h1)
		#############################################################
		#if mlp or single LSTM
		# model = phrase_lstm.TEST_MODEL(config.emb_size,config.pos_emb_size,config.tone_emb_size,
		# 	cons_num,vowel_num,vowel_ch_num,pretone_num,tone_num,postone_num,feat_num,phrase_num,dep_num,config.voc_size,pos_num,pos_feat_num,
		# 	config.lstm_hidden_size,config.f0_dim,config.linear_h1)
		#############################################################
		##if syllable lstm
		# model = phrase_lstm.SYL_LSTM(config.emb_size,config.pos_emb_size,config.tone_emb_size,
		# 	cons_num,vowel_num,vowel_ch_num,pretone_num,tone_num,postone_num,feat_num,phrase_num,dep_num,config.voc_size,pos_num,pos_feat_num,
		# 	config.lstm_hidden_size,config.f0_dim,config.linear_h1)
		#############################################################
		learning_rate = config.learning_rate
		param_list = []
		for name,param in model.named_parameters():
			if not param.requires_grad:
				print(name+" no gradient")
				continue
			if name in ["embed.weight","pos_embed.weight","cons_embed.weight","vowel_embed.weight"]:
				# print(name)
				param_list.append({'params': param, 'lr': learning_rate,"my_name":name})
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
		# seq2seq.Train(
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
			train_dep,
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
			test_dep,
			test_f0,
			test_len,
			model,
			optimizer,
			learning_rate,
			decay_step,
			decay_rate,
			epoch_num)



















