import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from scipy.fftpack import idct, dct
class MLP(nn.Module):
	def __init__(self,feature_number,out_size):
		super(MLP, self).__init__()
		self.hidden_unit = 50
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