import numpy as np

def ch_label(label,label_str = ["'setosa'","'versicolor'","'virginica'"]):
    data_point = len(label)
    # you should get this parameter correctly // 
    label_num = np.ones([data_point,1])
    ## your code here
    for i in range(data_point):
        if(label[i]==label_str[0]):
            label_num[i] = 0
        elif(label[i]==label_str[1]):
            label_num[i] = 1
        else:
            label_num[i] = 2
    ## end
    return label_num


## feature_nomalization 10 point
def feature_normalization(data, label): # *** Change from (data) to (data, label)
    # parameter 
    feature_num = data.shape[1]
    data_point = data.shape[0]
    # you should get this parameter correctly
    normal_feature = np.zeros([data_point,feature_num])
    ## your code here
    mu, sigma = get_normal_parameter(data, label, 3)
    for i, one_data in enumerate(data):
        _label = int(label[i])
        normal_feature[i,:] = ((one_data[:] - mu[_label][:]) / sigma[_label][:])

    ## end
    return normal_feature
        
def spilt_data(data,label,spilt_factor):
    # you should get this parameter correctly
    feature_num = data.shape[1]
    data_point = data.shape[0]
    train_data = np.zeros([spilt_factor,feature_num])
    train_label = np.zeros([spilt_factor,1])
    test_data = np.zeros([data_point - spilt_factor,feature_num])
    test_label = np.zeros([data_point - spilt_factor,1])
    ## your code here    
    train_num = [i for i in range(spilt_factor)]
    test_num = [i for i in range(spilt_factor,len(label))]
    train_data = data[train_num,:]
    test_data = data[test_num,:]
    train_label = label[train_num]
    test_label = label[test_num]
    ## end
    return train_data,test_data,train_label,test_label

## get_nomal_parameter 20 point
def get_normal_parameter(train_data,train_label,label_num):
    ## parameter
    feature_num = train_data.shape[1]
    ## you should get this parameter correctly    
    mu = np.zeros([label_num,feature_num])
    sigma = np.zeros([label_num,feature_num])
    ## your code here

    for label in range(label_num):
        one_label = []
 
        for i, one_data in enumerate(train_data):
            if(int(train_label[i]) == label):
                one_label = np.append(one_label, one_data)
 
        if np.size(one_label) != 0:
            one_label = one_label.reshape(int(np.size(one_label)/feature_num) , feature_num)
            for feature in range(feature_num):
                mu[label][feature] = np.mean(one_label[:,feature], dtype=np.float64)
                sigma[label][feature] = np.std(one_label[:,feature], dtype=np.float64)
      
        # If current label doesn't exist in data, set (mean,variance) to inf                  
        else:
            mu[label][:] = np.inf
            sigma[label][:] = np.inf
    
    ## end
    return mu,sigma

## prob 50 point
def prob(mu,sigma,data,label, prior = [0.33333333,0.33333333,0.33333333]):
    ## parameter
    data_point = data.shape[0]
    label_num = mu.shape[0]
    ## you should get this parameter correctly   
    prob = np.zeros([data_point,label_num])
    pi = np.zeros([label_num,1])
    ## your code here
    for i, one_data in enumerate(data):
        pi[int(label[i])] += 1
        for _feature, val in enumerate(one_data):
            for _label in range(label_num):
                _mu = mu[_label][_feature]
                _sigma = sigma[_label][_feature]

                if _mu != np.inf and _sigma != np.inf:
                    if _sigma == 0:
                        _sigma = 1e-9
                    exp = np.exp(-0.5*((val - _mu)/_sigma)**2)
                    likelihood = exp / (_sigma*np.sqrt(2*np.pi))
                    if likelihood == 0:
                        prob[i][_label] += -np.inf
                    else:
                        prob[i][_label] += np.log(likelihood)
                else:
                    prob[i][_label] += -np.inf

    for _label in range(label_num):
        pi[_label] /= data_point 
    for _label in range(label_num):
        prob[:,_label] += np.log(pi[_label])

    ## end
    return prob,pi

## classifier 20 point
def classifier(prob):
    ## parameter
    data_point = prob.shape[0]
    ## you should get this parameter correctly 
    label = np.zeros([data_point])
    ## your code here
    for i, _posterior in enumerate(prob):
        label[i] = np.argmax(_posterior)
        
    ## end
    return label
        
def acc(est,gnd):
    ## parameter
    total_num = len(gnd)
    ## you should get this parameter correctly 
    acc = 0
    ## your code here
    for i in range(total_num):
        if(est[i]==gnd[i]):
            acc = acc + 1
        else:
            acc = acc
    ## end
    return (acc / total_num)*100, acc