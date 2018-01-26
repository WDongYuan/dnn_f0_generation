import numpy as np
from sets import Set

def rmse(a,b):
	return np.sqrt(np.square(a-b).mean(axis=1)).mean()
def normalize(arr):
	mean = arr.mean(axis=1).reshape((arr.shape[0],1))
	std = (arr.std(axis=1)+0.0000001).reshape((arr.shape[0],1))
	shape = (arr-mean)/std
	return shape,mean,std


if __name__=="__main__":
	a = np.loadtxt("../pos_lstm_dev_res",delimiter=" ")
	b = np.zeros(a.shape)
	print(rmse(a,b))

	# tagset = Set([])
	# with open("../lstm_data/txt_token_pos") as f:
	# 	lines = f.readlines()
	# 	for i in range(len(lines)):
	# 		if i%4==2:
	# 			for tag in lines[i].strip().split(" "):
	# 				tagset.add(tag)
	# 				# if tag=="ON":
	# 				# 	print(lines[i-1])
	# 				# 	print(lines[i])
	# tagset = sorted(list(tagset))
	# # for tag in tagset:
	# # 	print(tag)
	# dic = {}
	# dic["AD"] = "AD"
	# dic["AS"] = "AS"
	# dic["BA"] = "BB"#bai bei
	# dic["CC"] = "CON"#CONJUCTION
	# dic["CD"] = "CD"
	# dic["CS"] = "CON"
	# dic["DEC"] = "DE"
	# dic["DEG"] = "DE"
	# dic["DER"] = "DE"
	# dic["DEV"] = "DE"
	# dic["DT"] = "DT"
	# dic["ETC"] = "ETC"
	# dic["IJ"] = "IJ"
	# dic["JJ"] = "JJ"
	# dic["LB"] = "BB"
	# dic["LC"] = "LC"
	# dic["M"] = "M"
	# dic["MSP"] = "CON"
	# dic["NN"] = "NN"
	# dic["NR"] = "NN"
	# dic["NT"] = "NN"
	# dic["OD"] = "CD"
	# dic["ON"] = "JJ"
	# dic["P"] = "P"
	# dic["PN"] = "PN"
	# dic["PU"] = "PU"
	# dic["SB"] = "BB"
	# dic["SP"] = "SP"
	# dic["VA"] = "JJ"
	# dic["VC"] = "VV"
	# dic["VE"] = "VV"
	# dic["VV"] = "VV"

	# for key,value in dic.items():
	# 	print(key),
	# 	print(value)
	
	