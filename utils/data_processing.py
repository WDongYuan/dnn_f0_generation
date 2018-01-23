cuda_flag = None
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.fftpack import idct, dct
# from stanfordcorenlp import StanfordCoreNLP
class EncodeFeature():
	# syl_numphones 0 0
	# p.syl_numphones 1 1
	# n.syl_numphones 2 2
	# stress 3 7
	# p.stress 8 13
	# n.stress 14 19
	# pos_in_word 20 20
	# ssyl_in 21 21
	# ssyl_out 22 22
	# syl_out 23 23
	# syl_in 24 24
	# asyl_in 25 25
	# asyl_out 26 26
	# next_accent 27 27
	# last_accent 28 28
	# accented 29 30
	# p.accented 31 32
	# n.accented 33 34
	# syl_endpitch 35 35
	# syl_startpitch 36 36
	# p.syl_endpitch 37 37
	# n.syl_startpitch 38 38
	# syl_break 39 43
	# p.syl_break 44 48
	# n.syl_break 49 53
	# syl_accent 54 56
	# p.syl_accent 57 59
	# n.syl_accent 60 62
	# R:SylStructure.parent.R:Word.content_words_in 63 63
	# R:SylStructure.parent.R:Word.content_words_out 64 64
	# R:SylStructure.parent.R:Word.pos_in_phrase 65 65
	# R:SylStructure.parent.R:Word.words_out 66 66
	# in_pos 67 67
	# out_pos 68 68
	# in_percent 69 69
	# tone 70 74
	def __init__(self,desc):
		self.encode_dic = []
		self.feature_pos = []
		with open(desc) as f:
			cont = f.readlines()
			cont = [line.strip()[1:-1] for line in cont]
			cont.pop(0)
			#delete the word.name feature
			cont.pop(0)
			cont.pop(-1)
		feat_pos = 0
		for feat in cont:
			feat = feat.split(" ")
			if feat[-1]=="float":
				self.encode_dic.append({})
				self.feature_pos.append([feat[0],[feat_pos,feat_pos]])
				feat_pos += 1
			else:
				tmp_dic = {}
				for i in range(1,len(feat)):
					tmp_dic[feat[i]] = i-1
				self.encode_dic.append(tmp_dic)
				self.feature_pos.append([feat[0],[feat_pos,feat_pos+len(tmp_dic)-1]])
				feat_pos += len(tmp_dic)

		print("feature position:")
		for tup in self.feature_pos:
			print(tup[0]+":"),
			print(tup[1][0]),
			print(tup[1][1])

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

def convert_feature(feat_file,f0_file,encode_feature,out_file,normalize_flag = False):
	f0 = np.loadtxt(f0_file,delimiter=" ")
	f0_mean = None
	f0_std = None
	if normalize_flag:
		f0_mean = f0.mean(axis=0)
		f0_std = f0.std(axis=0)
		f0 = (f0-f0_mean)/(f0_std+0.000001)
	with open(feat_file) as featf, open(out_file,"w+") as outf:
		feat = featf.readlines()
		assert len(feat)==len(f0)
		for i in range(len(f0)):
			outf.write(" ".join(f0[i].astype(np.str).tolist())+" "+encode_feature.encode(feat[i].strip())+"\n")
	return f0_mean,f0_std
	# with open(feat_file) as featf, open(f0_file) as f0f, open(out_file,"w+") as outf:
	# 	feat = featf.readlines()
	# 	f0 = f0f.readlines()
	# 	assert len(feat)==len(f0)
	# 	for i in range(len(f0)):
	# 		outf.write(f0[i].strip()+" "+encode_feature.encode(feat[i].strip())+"\n")

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

def get_f0_dct(utt_f0,utt_len,dct_num,noramlize_flag=False):
	utt_dct = np.zeros((utt_f0.shape[0],dct_num))
	# print(utt_f0.shape)
	# print(utt_dct.shape)
	for utt_i in range(len(utt_f0)):
		tmp_f0 = utt_f0[utt_i,0:utt_len[utt_i],:].flatten()
		tmp_dct = dct(tmp_f0)

		# print(tmp_dct.shape)
		# print(utt_dct.shape)
		utt_dct[utt_i,:] = tmp_dct[0:dct_num]
	dct_mean = None
	dct_std = None
	if noramlize_flag:
		dct_mean = utt_dct.mean(axis=0)
		dct_std = utt_dct.std(axis=0)
		utt_dct = (utt_dct-dct_mean)/(dct_std+0.00000001)
	return utt_dct,dct_mean,dct_std


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

def get_pos_dic(pos_file):
	pos_dic = {}
	with open(pos_file) as f:
		cont = f.readlines()
		for i in range(len(cont)):
			if i%4!=2:
				continue
			pos_l = cont[i].strip().split(" ")
			pos_l.pop(-1)
			for pos in pos_l:
				if pos not in pos_dic:
					pos_dic[pos] = len(pos_dic)+1
	return pos_dic
def append_pos_to_feature(feat_dir,pos_file,pos_dic):
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
	# pos_dic = {}
	# for key,val in data_dic.items():
	# 	for pos in val:
	# 		if pos not in pos_dic:
	# 			pos_dic[pos] = len(pos_dic)+1
	# pos_num = len(pos_dic)+1
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
	return

def one_hot_to_index(arr,zero_padding=True):
	new_arr = np.zeros((arr.shape[0],))
	for i in range(len(arr)):
		assert arr[i].sum()==1 or arr[i].sum()==0
		for j in range(len(arr[i])):
			if arr[i][j]==1:
				if zero_padding:
					new_arr[i] = j+1
				else:
					new_arr[i] = j
	return new_arr.astype(np.int32)

def get_syl_dic():
	cons_l = ["b","d","t","g","j","k","p","q","f","h","sh","s","ch","c","x","zh","z","m","n","l","r","w","y"]
	vowel_l = ["a","ai","ao","an","ang","o","ou","e","ei","en","eng","er","i","ia","iao","ian","iang","ie",
	"iu","in","ing","iong","iou","u","ua","uo","uai","uei","ui","uan","uen","uang","ueng","un","ong","v","ue"]
	c_dic = {}
	v_dic = {}
	for cons in cons_l:
		c_dic[cons] = len(c_dic)+1
	for vowel in vowel_l:
		v_dic[vowel] = len(v_dic)+1
	return c_dic,v_dic
def decompose_zh_syl(syl_l,c_dic,v_dic):
	result = []
	for syl in syl_l:
		if syl in v_dic:
			result.append([0,v_dic[syl]])
		else:
			p = 0
			while syl[0:p+1] in c_dic:
				p += 1
			# print(syl[0:p]+" "+syl[p:])
			result.append([c_dic[syl[0:p]],v_dic[syl[p:]]])
	result = np.array(result).astype(np.int32)
	return result
def append_syl_to_feature(feat_dir,map_file,c_dic,v_dic):
	data = {}
	with open(map_file) as f:
		for line in f:
			data_name,syl = line.strip().split(" ")
			if data_name not in data:
				data[data_name] = []
			data[data_name].append(syl[0:-1])
	file_list = os.listdir(feat_dir)
	for data_name in file_list:
		if "data" not in data_name:
			continue
		syl_l = data[data_name]
		cvl = decompose_zh_syl(syl_l,c_dic,v_dic)
		feat_cont = None
		with open(feat_dir+"/"+data_name) as f:
			feat_cont = f.readlines()
		feat_cont = [line.strip() for line in feat_cont]
		assert len(feat_cont)==len(cvl)
		with open(feat_dir+"/"+data_name,"w+") as f:
			for i in range(len(feat_cont)):
				feat_cont[i] = feat_cont[i]+" "+str(cvl[i][0])+" "+str(cvl[i][1])+"\n"
			f.writelines(feat_cont)
def normalize(arr):
	arr_mean = arr.mean(axis=0)
	arr_std = arr.std(axis=0)
	return (arr-arr_mean)/(0.0000001+arr_std),arr_mean,arr_std

def append_phrase_to_feature(feat_dir,phrase_syl_dir):
	file_list = os.listdir(feat_dir)
	for file in file_list:
		if "data" not in file:
			continue
		with open(phrase_syl_dir+"/"+file) as f:
			utt = f.readlines()
			for i in range(len(utt)):
				utt[i] = utt[i].strip().split(" ")

			phrase_feat = []
			for i in range(len(utt)):
				phrase = utt[i]
				for j in range(len(phrase)):
					word_feat = []
					#phrase position in utt
					word_feat.append(i)

					#phrase percent in utt
					word_feat.append(float(i)/len(utt))

					#phrase number in utt
					word_feat.append(len(utt))

					#syllable position in phrase
					word_feat.append(j)

					#syllable percent in phrase
					word_feat.append(float(j)/len(phrase))

					#syllable number in phrase
					word_feat.append(len(phrase))

					phrase_feat.append(word_feat)
			phrase_feat = np.array(phrase_feat)

			ori_feat = None
			with open(feat_dir+"/"+file) as featf:
				ori_feat = featf.readlines()
				assert len(phrase_feat)==len(ori_feat)
			with open(feat_dir+"/"+file,"w+") as outf:
				for i in range(len(ori_feat)):
					outf.write(ori_feat[i].strip()+" "+" ".join(phrase_feat[i].astype(np.str).tolist())+"\n")

def get_word_mean(emb,f0,voc_size,f0_mean=None):
	#get the mean f0 of the specific word for every f0 sample

	#emb 1 dimension
	#f0 2 dimension
	if f0_mean is None:
		print("is none")
		f0_dim = f0.shape[1]
		f0_sum = np.zeros((voc_size,f0_dim))
		f0_count = np.zeros((voc_size,1))
		for i in range(len(emb)):
			f0_sum[emb[i]] += f0[i]
			f0_count[emb[i]] += 1
		f0_mean = f0_sum/(f0_count+0.000001)

	new_f0 = np.zeros(f0.shape)
	for i in range(len(emb)):
		new_f0[i] = f0_mean[emb[i]]
	return new_f0,f0_mean











if __name__=="__main__":
	dic = word2index("../../mandarine/txt.done.data-all")
	


