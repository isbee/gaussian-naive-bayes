import numpy as np
import lsh as yms

# data logging
data = np.genfromtxt('data.csv')
label_str = np.genfromtxt('label.csv', dtype='unicode')
label = np.array(yms.ch_label(label_str))

# data shuffle
idx_shuffle = np.random.permutation(len(label))
label = label[idx_shuffle]
data = data[idx_shuffle,:]

# feature normalization (10 point)
normal_data = yms.feature_normalization(data, label)

# spilt data for testing
spilt_factor = 101
train_data,test_data,train_label,test_label = yms.spilt_data(data,label,spilt_factor)

# get train parameter of nomal distribution (20 point)
mu_train, sigma_train = yms.get_normal_parameter(train_data,train_label,3)

# get nomal distribution probability of each feature based on train feature (50 point)
prob,pi = yms.prob(mu_train,sigma_train,test_data,test_label)

# classification using prob (20 point)
estimation = yms.classifier(prob)

# get accuracy
acc, acc_s = yms.acc(estimation,test_label)

# print result
print('accuracy is ' + str(acc) + '% ! ! ')
print('the number of correct data is ' + str(acc_s) + ' of ' + str(len(test_label)) + ' ! ! ')