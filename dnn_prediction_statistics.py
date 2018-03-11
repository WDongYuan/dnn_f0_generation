import os
import argparse
import numpy as np
import random
from utils.data_processing import read_dic

def generate_f0_time_line(in_file,out_dir,true_f0_dir):
	out_file = out_dir+"/f0_timeline"
	os.system("mkdir "+out_file)
	os.system("python ../DCT/generate_f0_timeline.py"+
		" --mode generate_f0_timeline"+
		" --true_f0_dir "+true_f0_dir+
		" --predict_dir "+in_file+
		" --out_dir "+out_file+
		" --file_suffix .f0")
	out_file = out_file+"/f0_val"
	print("f0 timeline is saved to "+out_file)
	return out_file

def gen_txt_from_map(map_file,out_file,small_data = -1):
	dic = {}
	with open(map_file) as f:
		for line in f:
			line = line.strip().split(" ")
			data_name = line[0]
			syl = line[1]
			if data_name not in dic:
				dic[data_name] = []
			dic[data_name].append(syl)

	with open(out_file,"w+") as f:
		data_l = []
		for data_name,syl_l in dic.items():
			data_l.append([data_name,syl_l])
		data_l = sorted(data_l,key=lambda tup:tup[0])
		if small_data>0:
			data_l = data_l[0:small_data]
		for data_name,syl_l in data_l:
			f.write("( "+data_name+" \" "+" ".join(syl_l)+" \" )\n")

def test_statistics(predict_file,true_file,out_dir):
	os.system("python ../decision_tree/wagon/run.py"+
		" --mode predict_statistic"+
		" --file1 "+predict_file+
		" --file2 "+true_file+
		" > "+out_dir+"/"+"test_statistics")
	print("saving test statistics to "+out_dir+"/"+"test_statistics")

def put_predict_f0_in_file(in_file,syllable_map,out_dir,first_syllable_flag=1):
	out_file = out_dir+"/predict_f0_in_file"
	os.system("mkdir "+out_file)
	os.system("python ../decision_tree/wagon/run.py"+
		" --mode put_back_f0_in_file"+
		" --f0_file_map "+syllable_map+
		" --f0_val "+in_file+
		" --first_syllable_flag "+str(first_syllable_flag)+
		" --out_dir "+out_file)
	print("saving result to "+out_file)
	return out_file

def generate_f0_time_line(in_file,out_dir,true_f0_dir):
	out_file = out_dir+"/f0_timeline"
	os.system("mkdir "+out_file)
	os.system("python ../DCT/generate_f0_timeline.py"+
		" --mode generate_f0_timeline"+
		" --true_f0_dir "+true_f0_dir+
		" --predict_dir "+in_file+
		" --out_dir "+out_file+
		" --file_suffix .f0")
	out_file = out_file+"/f0_val"
	print("f0 timeline is saved to "+out_file)
	return out_file

def generate_wav(in_file,ccoef_dir,out_dir,smooth=0):
	prefix_name = "/".join(in_file.split("/")[0:-1])
	os.system("python ../test_f0_in_system/run.py"+
		" --mode synthesis_with_f0"+
		" --awb_synth ../test_f0_in_system/my_synth_f0"+
		" --ccoef_dir "+ccoef_dir+
		" --f0_dir "+in_file+
		" --out_dir "+prefix_name+"/wav"+
		" --smooth "+str(smooth))
	out_file = prefix_name+"/wav"
	print("waveform files saved to "+out_file)
	return out_file

def timeline_statistics(predict_f0,true_f0):
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode timeline_predict_statistics"+
		" --true_f0_dir "+true_f0+
		" --predict_f0_dir "+predict_f0)

def idct_phrase(phrase_syllable_dir,dct_dir,out_dir):
	out_dir += "/idct_phrase_f0"
	os.system("mkdir "+out_dir)
	os.system("python ../decision_tree/data_preprocessing.py"+
		" --mode idct_phrase_f0_dir"+
		" --phrase_syl_dir "+phrase_syllable_dir+
		" --dct_dir "+dct_dir+
		" --out_dir "+out_dir)
	return out_dir

def map_f0(f0_dir,map_file,out_file):
	os.system("python ../decision_tree/data_preprocessing.py"+
		" --mode map_to_new_f0_vector"+
		" --map_file "+map_file+
		" --f0_dir "+f0_dir+
		" --out_file "+out_file+
		" --add_index_prefix 0")



if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode',dest='mode')
	parser.add_argument('--voice_dir', dest='voice_dir')
	parser.add_argument('--operation_file', dest='op_file',default="./operation.desc")
	parser.add_argument('--train_data',dest='train_data')
	parser.add_argument('--train_label',dest='train_label')
	parser.add_argument('--dev_data',dest='dev_data')
	parser.add_argument('--dev_label',dest='dev_label')
	parser.add_argument('--out_dir',dest='out_dir')
	parser.add_argument('--train_syllable_map',dest='train_syllable_map')
	parser.add_argument('--dev_syllable_map',dest='dev_syllable_map')
	parser.add_argument('--vector_feat_desc',dest='vector_feat_desc')
	parser.add_argument('--val_feat_desc',dest='val_feat_desc')
	parser.add_argument('--true_f0_dir',dest='true_f0_dir')
	parser.add_argument('--ccoef_dir',dest='ccoef_dir')
	parser.add_argument('--data_dir',dest='data_dir')
	parser.add_argument('--predict_file',dest='predict_file')
	parser.add_argument('--predict_file_map',dest='predict_file_map')
	parser.add_argument('--txt_file',dest='txt_file')
	parser.add_argument('--dur_prediction',dest='dur_prediction',type=int,default=0)
	args = parser.parse_args()
	
	mode = args.mode
	if mode=="how_to_run":
		print("python dnn_prediction_statistics.py --mode stat"+
			" --voice_dir"+
			" --data_dir"+
			" --predict_file")
		print("python dnn_prediction_statistics.py"+
			" --mode generate_predicted_wav"+
			" --voice_dir"+
			" --data_dir"+
			" --predict_file"+
			" --predict_file_map"+
			" --dur_prediction"+
			" --out_dir")
		print("python dnn_prediction_statistics.py"+
			" --mode generate_syllable_test_data"+
			" --txt_file ../experiment/test_line"+
			" --out_dir self_generated_test_data")
	elif mode=="stat":
		voice_dir = args.voice_dir
		# out_dir = args.out_dir
		op_file = args.op_file
		data_dir = args.data_dir
		train_data = data_dir+"/train_test_data/train_data/train_feat"
		train_label = data_dir+"/train_test_data/train_data/train_f0"
		dev_data = data_dir+"/train_test_data/test_data/test_feat"
		dev_label = data_dir+"/train_test_data/test_data/test_f0"
		train_syllable_map = data_dir+"/train_test_data/train_data/train_syllable_map"
		dev_syllable_map = data_dir+"/train_test_data/test_data/test_syllable_map"
		true_f0_dir = data_dir+"/f0_value"
		ccoef_dir = voice_dir+"/ccoefs"
		vector_feat_desc = data_dir+"/new_feature_desc_vector"
		val_feat_desc = data_dir+"/new_feature_desc_val"
		predict_f0 = args.predict_file

		out_dir = "_tmp_out"
		os.system("mkdir "+out_dir)

		######################################################################
		print("")
		print(">>>>>>>>>> test statistics <<<<<<<<<<")
		test_statistics(predict_f0,dev_label,out_dir)
		os.system("cat "+out_dir+"/"+"test_statistics")
		######################################################################


		######################################################################
		print("")
		print(">>>>>>>>>> put f0 back in file <<<<<<<<<<")
		predict_f0 = put_predict_f0_in_file(predict_f0,dev_syllable_map,out_dir)
		f0_in_file = predict_f0
		######################################################################


		######################################################################
		print("")
		print(">>>>>>>>>> generate f0 timeline <<<<<<<<<<")
		predict_f0 = generate_f0_time_line(predict_f0,out_dir,true_f0_dir)
		f0_timeline_dir = predict_f0
		######################################################################

		######################################################################
		print("")
		print(">>>>>>>>>> predict statistics for f0 timeline <<<<<<<<<<")
		predict_f0 = timeline_statistics(predict_f0,true_f0_dir)
		######################################################################

		os.system("rm -r "+out_dir)

	elif mode=="generate_predicted_wav":
		out_dir = args.out_dir
		voice_dir = args.voice_dir
		# txt_file = args.txt_file
		predict_file = args.predict_file
		predict_file_map = args.predict_file_map
		dur_prediction = args.dur_prediction
		data_dir = args.data_dir

		os.system("rm -r "+out_dir)
		os.system("mkdir "+out_dir)

		#Creat the txt.done.data file from the syllable map
		txt_file = out_dir+"/tmp_txt"
		gen_txt_from_map(predict_file_map,txt_file,small_data=10)

		print("")
		print(">>>>>>>>>> put f0 back in file <<<<<<<<<<")
		predict_f0 = put_predict_f0_in_file(predict_file,predict_file_map,out_dir)
		f0_in_file = predict_f0

		################################################
		#create a small set of data for testing
		file_set = set()
		with open(txt_file) as f:
			for line in f:
				file_set.add(line.split(" ")[1])
		small_f0_in_file = f0_in_file+"_small"
		os.system("mkdir "+small_f0_in_file)
		for file in file_set:
			os.system("ln "+f0_in_file+"/"+file+" "+small_f0_in_file+"/"+file)
		f0_in_file = small_f0_in_file
		################################################


		if dur_prediction==0:
			true_f0_dir = data_dir+"/f0_value"
			ccoef_dir = voice_dir+"/ccoefs"
			
		elif dur_prediction==1:
			print(">>>>>>>>>> apply f0 in festival prediction <<<<<<<<<<")
			os.system("nohup python ../seq_op/apply_f0.py"+
				" --mode run"+
				" --voice_dir "+voice_dir+
				" --out_dir "+out_dir+
				" --modified_clustergen_scm /Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/src/experiment/clustergen.scm"+
				" --test_txt "+txt_file)
			ccoef_dir = out_dir+"/ccoefs"
			f0_tag_dir = out_dir+"/f0_value"
			true_f0_dir = f0_tag_dir

		################################################
		#modify the f0 here
		file_to_modify = os.listdir(f0_in_file)
		syl_to_modify = []
		idx_to_modify = [8,9]
		def modify_operation(arr):
			mean = arr.mean()
			std = arr.std()
			norm = (arr-mean)/std
			return norm*std*2+mean

		lines = None
		for mod_file in file_to_modify:
			with open(f0_in_file+"/"+mod_file) as f:
				lines = f.readlines()
				lines = [line.strip().split(" ") for line in lines]
				for li in range(len(lines)):
					line = lines[li]
					if line[0] in syl_to_modify:
						lines[li] = [line[0]]+modify_operation(np.array(line[1:]).astype(np.float)).astype(str).tolist()
					if li in idx_to_modify:
						lines[li] = [line[0]]+modify_operation(np.array(line[1:]).astype(np.float)).astype(str).tolist()
			with open(f0_in_file+"/"+mod_file,"w+") as f:
				for line in lines:
					f.write(" ".join(line)+"\n")
		################################################

		predict_f0 = generate_f0_time_line(f0_in_file,out_dir,true_f0_dir)

		print("")
		print(">>>>>>>>>> generate waveform files from f0 <<<<<<<<<<")
		predict_f0 = generate_wav(predict_f0,ccoef_dir,out_dir,smooth=0)
		print("waveform files saved to "+predict_f0)



	elif mode=="generate_syllable_test_data":

		txt_file = args.txt_file
		test_dir = args.out_dir

		def tmp_decompose_zh_syl(syl_l,c_dic,v_dic,v_c_dic):
			result = []
			for syl in syl_l:
				split_l = []
				tone = int(syl[-1])
				syl = syl[0:-1]
				vowel = ""
				if syl in v_dic:
					split_l += [0,v_dic[syl]]
					vowel = syl
					# result.append([0,v_dic[syl]])
				else:
					p = 0
					while syl[0:p+1] in c_dic:
						p += 1
					vowel = syl[p:]
					split_l += [syl[0:p],syl[p:]]
				split_l.append(tone)
				result.append(split_l)
			return result

		data = []
		# with open("../experiment/txt.done.data.test") as f:
		with open(txt_file) as f:
			for line in f:
				line = line.strip().split(" ")
				data_name = line[1]
				syl_l = line[3:-2]
				line_syl = tmp_decompose_zh_syl(syl_l,read_dic("dic_dir/cons_dic"),read_dic("dic_dir/vowel_dic"),read_dic("dic_dir/vowel_char_dic"))
				for syl in line_syl:
					syl.append(data_name)
				data.append(line_syl)

		cons_dic = read_dic("dic_dir/cons_dic")
		vowel_dic = read_dic("dic_dir/vowel_dic")
		tone_dic = {1: 0, 3: 1, 2: 2, 5: 3, 4: 4}
		feat = np.zeros((len(data),len(data[0]),156))##10 value for f0 value
		for i in range(len(data)):
			for j in range(len(data[i])):
				if data[i][j][0]!="":
					feat[i][j][84] = cons_dic[data[i][j][0]]
				if data[i][j][1]!="":
					feat[i][j][85] = vowel_dic[data[i][j][1]]
				feat[i][j][13+tone_dic[data[i][j][2]]] = 1

		os.system("mkdir "+test_dir)
		os.system("mkdir "+test_dir+"/data_dir")

		#Create the map file for the self generated data
		with open(test_dir+"/data_map","w+") as f:
			for i in range(len(feat)):
				# data_name = "data_"+"".join(["0" for j in range(5-len(str(i+1)))])+str(i+1)
				data_name = data[i][0][-1]
				np.savetxt(test_dir+"/data_dir/"+data_name,feat[i],delimiter=" ",fmt="%.5f")

				for tup in data[i]:
					# print(tup)
					f.write(data_name+" "+tup[0]+tup[1]+str(tup[2])+"\n")



