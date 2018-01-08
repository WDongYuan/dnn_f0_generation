cuda_flag = True

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
# from stanfordcorenlp import StanfordCoreNLP
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

def word2index(txt_file,voc_size):
	##return a dictionary for word-count
	dic = {}
	with open(txt_file) as f:
		for line in f:
			line = line.split(" ")[2].decode("utf-8")[1:-1]
			for word in line:
				word = word.encode("utf-8")
				if word not in dic:
					dic[word] = 0
				dic[word] += 1

	word_list = []
	for word,count in dic.items():
		word_list.append([word,count])
	word_list = sorted(word_list,key=lambda tup: tup[1],reverse=True)

	####################################################
	##the first 1500 words cover 91% of the corpus
	# all_count = 0
	# for word,count in word_list:
	# 	all_count += count
	# tmp_count = 0
	# for word,count in word_list:
	# 	tmp_count += count
	# 	print(word+" "+str(float(tmp_count)/all_count))
	####################################################

	dic = {}
	cut_thr = voc_size
	for i in range(1,cut_thr-1):
		dic[word_list[i-1][0]] = i
	dic["UNK"] = cut_thr-1
	# for word,index in dic.items():
	# 	print(str(index)+" "+word)
	return dic

def utt_content(txt_file):
	dic = {}
	with open(txt_file) as f:
		for line in f:
			content = line.split(" ")[2].decode("utf-8")[1:-1]
			data_name = line.split(" ")[1]
			dic[data_name] = content
	return dic

def collect_utt_data(data_file,map_file,out_dir,txt_file,word_index):
	os.system("mkdir "+out_dir)
	with open(data_file) as df, open(map_file) as mf:
		data_line = df.readlines()
		map_line = mf.readlines()
		assert len(data_line)==len(map_line)
		cur_utt = ""
		cur_file_cont = ""
		for i in range(len(data_line)):
			tmp_utt,tmp_syl = map_line[i].strip().split(" ")
			if tmp_utt!=cur_utt and cur_utt!="":
				with open(out_dir+"/"+cur_utt,"w+") as tmp_f:
					tmp_f.write(cur_file_cont)
				cur_file_cont = ""
			cur_utt = tmp_utt
			cur_file_cont += data_line[i]
		with open(out_dir+"/"+cur_utt,"w+") as tmp_f:
			tmp_f.write(cur_file_cont)

	utt_cont = utt_content(txt_file)
	utt_list = os.listdir(out_dir)
	for utt_file in utt_list:
		if "data" not in utt_file:
			continue
		with open(out_dir+"/"+utt_file) as f:
			cont = f.readlines()
		cont = [line.strip() for line in cont]

		utt_word = utt_cont[utt_file][0:-1]
		# print(utt_file)
		# print(len(cont))
		# print(len(utt_word))
		# print(utt_word)
		assert len(cont)==len(utt_word)
		for i in range(len(utt_word)):
			word = utt_word[i].encode("utf-8")
			if word not in word_index:
				cont[i] += " "+str(word_index["UNK"])
			else:
				cont[i] += " "+str(word_index[word])
		with open(out_dir+"/"+utt_file,"w+") as f:
			for line in cont:
				f.write(line+"\n")

	return


def create_train_val_data(data_file,train_ratio):
	data = np.loadtxt(data_file,delimiter=" ")
	np.random.shuffle(data)
	train_data = data[0:int(train_ratio*len(data)),10:]
	train_data = (train_data-train_data.mean(axis=0))/(train_data.std(axis=0)+0.0001)
	train_label = data[0:int(train_ratio*len(data)),0:10]
	val_data = data[int(train_ratio*len(data)):,10:]
	val_data = (val_data-val_data.mean(axis=0))/(val_data.std(axis=0)+0.0001)
	val_label = data[int(train_ratio*len(data)):,0:10]
	return train_data,train_label,val_data,val_label

def get_f0_embedding(data_dir):
	file_list = os.listdir(data_dir)
	embedding = []
	f0 = []
	max_length = 0
	for file in file_list:
		if "data" not in file:
			continue
		sent_word = []
		sent_f0 = []
		file_cont = np.loadtxt(data_dir+"/"+file,delimiter=" ")
		max_length = max(file_cont.shape[0],max_length)
		f0.append(file_cont[:,0:10])
		embedding.append(file_cont[:,-1])
	embedding_arr = np.zeros((len(embedding),max_length))
	f0_arr = np.zeros((len(f0),max_length,10))
	length_arr = np.zeros((len(embedding),))
	for i in range(len(embedding)):
		assert len(embedding[i])==len(f0[i])
		length_arr[i] = len(embedding[i])
		embedding_arr[i,0:len(embedding[i])] = embedding[i]
		f0_arr[i,0:len(f0[i]),:] = f0[i]
	return f0_arr,embedding_arr.astype(np.int64),length_arr.astype(np.int64)

def get_f0_feature(data_dir):
	file_list = os.listdir(data_dir)
	feature = []
	f0 = []
	max_length = 0
	feat_num = 0
	for file in file_list:
		if "data" not in file:
			continue
		file_cont = np.loadtxt(data_dir+"/"+file,delimiter=" ")
		max_length = max(file_cont.shape[0],max_length)
		f0.append(file_cont[:,0:10])
		feature.append(file_cont[:,10:])
		feat_num = file_cont.shape[1]-10
	feature_arr = np.zeros((len(feature),max_length,feat_num))
	f0_arr = np.zeros((len(f0),max_length,10))
	length_arr = np.zeros((len(feature),))
	for i in range(len(feature)):
		assert len(feature[i])==len(f0[i])
		length_arr[i] = len(feature[i])
		feature_arr[i,0:len(feature[i]),:] = feature[i]
		f0_arr[i,0:len(f0[i]),:] = f0[i]
	return f0_arr,feature_arr,length_arr.astype(np.int64)

def get_f0_feature_list(data_dir,feat_list):
	file_list = os.listdir(data_dir)
	feature = []
	f0 = []
	max_length = 0
	feat_num = 0
	for file in file_list:
		if "data" not in file:
			continue
		file_cont = np.loadtxt(data_dir+"/"+file,delimiter=" ")
		max_length = max(file_cont.shape[0],max_length)
		f0.append(file_cont[:,0:10])
		feature.append(file_cont[:,feat_list])
		feat_num = len(feat_list)
	feature_arr = np.zeros((len(feature),max_length,feat_num))
	f0_arr = np.zeros((len(f0),max_length,10))
	length_arr = np.zeros((len(feature),))
	for i in range(len(feature)):
		assert len(feature[i])==len(f0[i])
		length_arr[i] = len(feature[i])
		feature_arr[i,0:len(feature[i]),:] = feature[i]
		f0_arr[i,0:len(f0[i]),:] = f0[i]
	return f0_arr,feature_arr,length_arr.astype(np.int64)


def get_shape_mean_std(f0_arr,len_arr):
	shape_arr = np.zeros(f0_arr.shape)
	mean_arr = np.zeros((f0_arr.shape[0],f0_arr.shape[1],1))
	std_arr = np.zeros((f0_arr.shape[0],f0_arr.shape[1],1))
	for utt_id in range(f0_arr.shape[0]):
		utt_len = len_arr[utt_id]
		tmp_mean = f0_arr[utt_id,0:utt_len,:].mean(axis=1).reshape((-1,1))
		tmp_std = f0_arr[utt_id,0:utt_len,:].std(axis=1).reshape((-1,1))
		tmp_shape = (f0_arr[utt_id,0:utt_len,:]-tmp_mean)/(tmp_std+0.000001)
		shape_arr[utt_id,0:utt_len,:] = tmp_shape
		mean_arr[utt_id,0:utt_len,:] = tmp_mean
		std_arr[utt_id,0:utt_len,:] = tmp_std
	return shape_arr,mean_arr,std_arr

def parse_txt_file(txt_file,out_file):
	parser = StanfordCoreNLP(r'/Users/weidong/Downloads/stanford-corenlp-full-2016-10-31',lang='zh')
	with open(txt_file) as txtf, open(out_file,"w+") as outf:
		for line in txtf:
			line = line.strip().split(" ")
			data_name = line[1]
			text = line[2].decode("utf-8")[1:-1].encode("utf-8")
			tokens = parser.word_tokenize(text)
			pos = parser.pos_tag(text)
			outf.write(data_name+"\n")
			outf.write(" ".join(tokens).encode("utf-8")+"\n")
			outf.write(" ".join([tup[1] for tup in pos]).encode("utf-8")+"\n")
			outf.write("\n")
	return

def append_pos_to_feature(feat_dir,pos_file):
	##read pos tag
	data_dic = {}

	with open(pos_file,encoding='utf-8') if cuda_flag else open(pos_file) as f:
		sents = f.readlines()
		row = 0
		while row < len(sents):
			data_name = sents[row].strip()
			row += 1
			token = sents[row].strip().split(" ")
			row += 1
			pos = sents[row].strip().split(" ")
			row += 1
			row += 1
			token.pop(-1)
			pos.pop(-1)
			pos_list = []
			assert len(token)==len(pos)
			for i in range(len(token)):
				for j in range(len(token[i]) if cuda_flag else len(token[i].decode("utf-8"))):
					pos_list.append(pos[i])
			data_dic[data_name] = pos_list
			# if data_name=="data_00001":
			# 	print(pos_list)
			# 	for tk in token:
			# 		print(tk.encode("utf-8")),
			# 	print("")
	pos_dic = {}
	for key,val in data_dic.items():
		for pos in val:
			if pos not in pos_dic:
				pos_dic[pos] = len(pos_dic)+1
	pos_num = len(pos_dic)+1
	# print(pos_dic)

	file_list = os.listdir(feat_dir)
	file_list = [tmp_name for tmp_name in file_list if "data" in tmp_name]
	for file_name in file_list:
		pos = data_dic[file_name]
		file_sents = None
		with open(feat_dir+"/"+file_name) as f:
			file_sents = f.readlines()
		assert len(file_sents)==len(pos)
		file_sents = [file_sents[i].strip()+" "+str(pos_dic[pos[i]])+"\n" for i in range(len(file_sents))]
		with open(feat_dir+"/"+file_name,"w+") as f:
			f.writelines(file_sents)
	return pos_num







if __name__=="__main__":
	dic = word2index("../../mandarine/txt.done.data-all")
	


