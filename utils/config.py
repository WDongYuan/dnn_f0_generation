cuda_flag = True
update_data = False
dct_num = 5
dct_flag = False

learning_rate = 0.002
decay_step = 1
decay_rate = 0.96
epoch_num = 100
voc_size = 2500
batch_size = 20
emb_size = 300
pos_emb_size = 10
tone_emb_size = 10
lstm_hidden_size = 100
f0_dim = 10
linear_h1 = 200


#################
##embedding best
# learning_rate = 0.01
# decay_step = 10
# decay_rate = 0.3
# epoch_num = 100
# voc_size = 2500
# batch_size = 20
# emb_size = 20
# lstm_hidden_size = 100
# f0_dim = 10
# linear_h1 = 200
##loss 32.0


##feature best
# learning_rate = 0.01
# decay_step = 15
# decay_rate = 0.3
# epoch_num = 100
# voc_size = 2500
# batch_size = 20
# emb_size = 20
# lstm_hidden_size = 100
# f0_dim = 10
# linear_h1 = 150
#loss 31.1

##emb_mean_std
# learning_rate = 0.01
# decay_step = 10
# decay_rate = 0.3
# epoch_num = 100
# voc_size = 2500
# batch_size = 20
# emb_size = 20
# lstm_hidden_size = 50
# f0_dim = 10
# linear_shape = 100
# linear_mean = 50
# linear_std = 50

##emb_feat
# learning_rate = 0.001
# decay_step = 1
# decay_rate = 0.9
# epoch_num = 100
# voc_size = 2500
# batch_size = 20
# emb_size = 20
# pos_emb_size = 10
# lstm_hidden_size = 100
# f0_dim = 10
# linear_h1 = 100
#28.78


##emb_pos_feat
# learning_rate = 0.001
# decay_step = 1
# decay_rate = 0.95
# epoch_num = 100
# voc_size = 2500
# batch_size = 20
# emb_size = 20
# pos_emb_size = 10
# lstm_hidden_size = 100
# f0_dim = 10
# linear_h1 = 200
##27.7


##Tone LSTM
# cuda_flag = True
# learning_rate = 0.001
# decay_step = 1
# decay_rate = 0.95
# epoch_num = 100
# voc_size = 2500
# batch_size = 20
# emb_size = 10
# pos_emb_size = 10
# tone_emb_size = 10
# lstm_hidden_size = 100
# f0_dim = 10
# linear_h1 = 200
#27.3


##no tone, emb, feat, syllable, pos
# cuda_flag = False
# learning_rate = 0.001
# decay_step = 1
# decay_rate = 0.95
# epoch_num = 100
# voc_size = 2500
# batch_size = 20
# emb_size = 10
# pos_emb_size = 10
# tone_emb_size = 10
# lstm_hidden_size = 100
# f0_dim = 10
# linear_h1 = 200
#27.2

##emb_feat_pos_lstm+cons_vowel_tone_lstm
# cuda_flag = True
# learning_rate = 0.001
# decay_step = 1
# decay_rate = 0.95
# epoch_num = 100
# voc_size = 2500
# batch_size = 20
# emb_size = 10
# pos_emb_size = 10
# tone_emb_size = 10
# lstm_hidden_size = 100
# f0_dim = 10
# linear_h1 = 200
# 27.0

##emb_feat_pos_lstm+cons_vowel_tone_lstm_phrase
# learning_rate = 0.001
# decay_step = 1
# decay_rate = 0.95
# epoch_num = 100
# voc_size = 2500
# batch_size = 20
# emb_size = 10
# pos_emb_size = 10
# tone_emb_size = 10
# lstm_hidden_size = 100
# f0_dim = 10
# linear_h1 = 200
#26.7

##cnn mean
# learning_rate = 0.0005
# decay_step = 1
# decay_rate = 0.95
# epoch_num = 100
# voc_size = 2500
# batch_size = 20
# emb_size = 10
# pos_emb_size = 10
# tone_emb_size = 10
# lstm_hidden_size = 100
# f0_dim = 10
# linear_h1 = 100
#mean 19.0


##pos lstm
# learning_rate = 0.001
# decay_step = 1
# decay_rate = 0.95
# epoch_num = 100
# voc_size = 2500
# batch_size = 20
# emb_size = 10
# pos_emb_size = 10
# tone_emb_size = 10
# lstm_hidden_size = 100
# f0_dim = 10
# linear_h1 = 200
# 25.9

# 25.3
# feat_h_0 = torch.cat((emb,feat,pos,pos_feat,dep),dim=2)
# ph_h_0 = torch.cat((feat,cons,vowel,tone,phrase),dim=2)