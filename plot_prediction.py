import os
import sys
import random
if __name__=="__main__":
	if len(sys.argv)==1:
		print("python plot_prediction.py emb_pos_feat_prediction")
		exit()

	predict_file = sys.argv[1]
	os.system("mkdir dump")
	os.system("python ../decision_tree/wagon/run.py"+
		" --mode put_back_f0_in_file"+
		" --f0_file_map ../mandarine/gen_f0/train_dev_data_vector/dev_data/syllable_map"+
		" --f0_val "+predict_file+
		" --out_dir ./dump/predict_f0_in_file")
	os.system("python ../DCT/generate_f0_timeline.py"+
		" --mode generate_f0_timeline"+
		" --true_f0_dir ../mandarine/gen_f0/f0_value"+
		" --predict_dir ./dump/predict_f0_in_file"+
		" --out_dir ./dump/predict_f0_dir"+
		" --file_suffix .f0")

	gen_wav = raw_input("generate waveform file?(y/n)")
	if gen_wav=="y":
		os.system("python ../test_f0_in_system/run.py"+
			" --mode synthesis_with_f0"+
			" --awb_synth ../test_f0_in_system/my_synth_f0"+
			" --ccoef_dir ../mandarine/cmu_yue_wdy_cn/ccoefs"+
			" --f0_dir ./dump/predict_f0_dir/f0_val"+
			" --out_dir ./dump/predict_f0_dir/wav")

	file_list = os.listdir("./dump/predict_f0_dir/f0_val")
	data_name = ""
	while True:
		random.shuffle(file_list)
		print("data_name example:")
		print(file_list[0:20])
		data_name = raw_input("input data_name to plot(input \"end\" to end):")
		if data_name == "end":
			break
		os.system("python ../test_f0_in_system/run.py"+
			" --mode plot_two_file_with_syllable"+
			" --true_f0_dir ../mandarine/gen_f0/f0_value"+
			" --predict_dir ./dump/predict_f0_in_file"+
			" --dir1 ../test_f0_in_system/true_cn_f0/f0_val"+
			" --dir2 ./dump/predict_f0_dir/f0_val"+
			" --data_name "+data_name)
