import os
import argparse
import numpy as np

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

def generate_wav(in_file,ccoef_dir,out_dir):
	prefix_name = "/".join(in_file.split("/")[0:-1])
	os.system("python ../test_f0_in_system/run.py"+
		" --mode synthesis_with_f0"+
		" --awb_synth ../test_f0_in_system/my_synth_f0"+
		" --ccoef_dir "+ccoef_dir+
		" --f0_dir "+in_file+
		" --out_dir "+prefix_name+"/wav")
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
	parser.add_argument('--decompose_desc',dest='decompose_desc',default="./decompose_desc")
	parser.add_argument('--vector_feat_desc',dest='vector_feat_desc')
	parser.add_argument('--val_feat_desc',dest='val_feat_desc')
	parser.add_argument('--true_f0_dir',dest='true_f0_dir')
	parser.add_argument('--ccoef_dir',dest='ccoef_dir')
	parser.add_argument('--data_dir',dest='data_dir')
	parser.add_argument('--predict_file',dest='predict_file')
	args = parser.parse_args()
	
	mode = args.mode
	if mode=="how_to_run":
		print("python dnn_prediction_statistics.py --mode stat"+
			" --voice_dir"+
			" --data_dir"+
			" --predict_file"+
			" --out_dir")
	elif mode=="stat":
		voice_dir = args.voice_dir
		out_dir = args.out_dir
		op_file = args.op_file
		data_dir = args.data_dir
		decompose_desc = args.decompose_desc
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

		os.system("rm -r "+f0_in_file)
		# os.system("rm -r "+f0_timeline_dir)
