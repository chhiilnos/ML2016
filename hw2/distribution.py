import numpy as np
     
class decision_tree():
  
  feature = [[0 for x in range(4001)] for y in range(57)]
  train = []
  feature_max = [0.]*57
  classify = [0.]*4001

  def read_train(self):
    with open('spam_train.csv') as data_file:
      for line in data_file:
        self.train.append(line.strip().split(','))
    data_file.close()
    for i in (len(train)):
      for j in (len(train[0])):
        self.train[i][j]=int(self.train[i][j])
    for i in range (4001):
      for j in range (57):
        self.feature[j][i]=self.train[i][j+1]
      self.classify[i] = self.train[i][58]
    for i in range (57):
      for k in range(4001):
        if(self.feature[i][k]>self.feature_max[i]):
          self.feature_max[i]=self.feature[i][k]
    
  def read_test(self):
    with open('spam_test.csv') as data_file:
      for line in data_file:
        self.test.append(line.strip().split(','))
    data_file.close()
    for i in range (len(self.test)):
      for j in range (len(self.test[0])):
        self.test[i][j]=int(self.test[i][j])

  def information_gain(self,feature_index,input_data_index,threshold,entropy_before):##return left_data index,right_data_index,information_gain
    left_data_index = []
    right_data_index = []
    for i in range (len(input_data_index)):##classify for node with respect to threshold
      if (self.feature[int(feature_index)][int(input_data_index[i])]>float(threshold)*0.2*self.feature_max[int(feature_index)]):
        left_data_index.append(int(input_data_index[i]))
      else:
        right_data_index.append(int(input_data_index[i]))
    ##calculate entropy for left
    num_left_1 = 0
    for i in (len(left_data_index)):
      if (classify[left_data_index[i]]==1):
        num_left_1 = num_left_1+1
    left_p = float(num_left_1)/float(len(left_data_index))
    entropy_left = (-1)*(left_p*np.log(left_p)+(1.-left_p)*np.log(1.-left_p))
    
    ##calculate entropy for right
    num_right_1 = 0
    for i in (len(right_data_index)):
      if (self.classify[right_data_index[i]] == 1):
        num_right_1 = num_right_1 + 1
    right_p = float(num_right_1)/float(len(right_data_index))
    entropy_right = (-1.)*(right_p*np.log(right_p)+(1.-right_p)*np.log(1.-right_p))
    
    ##calculate entropy after
    entropy_after = (float(left_data_index)/float(input_data_index))*entropy_left+(float(right_data_index)/float(input_data_index))*entropy_right

    ##calculate entropy gain
    information_gain = float(entropy_after)-float(entropy_before)
    
    return(left_data_index,right_data_index,information_gain,entropy_after,entropy_left,entropy_right)

  ##decide threshold to get max information gain for each node
  def threshold_decision(self,feature_index,input_data_index,entropy_before):
    information_gain_max = 0.
    max_information_gain_threshold = 0
    for i in range(5):		
      if(information_gain(int(feature_index),input_data_index,int(threshold),int(entropy_before))>information_gain_max):
        information_gain_max = information_gain(int(feature_index),input_data_index,int(threshold),int(entropy_before))[2]
        max_information_gain_threshold=i
    return(max_information_gain_threshold)
  ###
  def feature_threshold_decision(input_data_index,entropy_before):
    information_gain_max_index = 0
    temp_information_gain_max = 0.
    for feature_index in range (57):
      if (threshold_decision(feature_index,input_data_index,entropy_before)>temp_information_gain_max):
        information_gain_max_index = feature_index
        temp_information_gain_max = threshold_decision(feature_index,input_data_index,entropy_before)
    return(information_gain_max_index,temp_information_gain_max)
    
  ##returns initial entropy
  def initial_entropy(self):
    num_class_1 = 0
    for k in range(4001):
      if (classify[k]==1):
        num_class_1=num_class_1+1
    p = float(num_class_1)/4001
    initial_entropy = (-1)*(p*np.log(p)+(1-p)*np.log(1-p))
    return(initial_entropy)
  
  def parent(node_index):
    return(int(node_index/2))
  
  

  ###grow a tree with layer n:
  node_feature = []
  node_threshold = []
  node_data_index = []
  node_entropy = []
  
  def _init_(self,node_1_feature,layer_num):
    read_train()
    read_test()
    self.node_data_index.append([0]*10)
    self.node_entropy.append(0.)
    self.node_feature.append(0)
    self.node_threshold.append(0.)
    self.node_data_index.append([l for l in range (4001)])
    self.node_entropy.append(self.initial_entropy())
    self.node_feature.append(int(node_1_feature))
    self.node_threshold.append(self.threshold_decision(node_1_feature,self.node_data_index[1],self.node_entropy[1]))
    self.node_entropy.append(self.information_gain(self.node_feature[1],self.node_data_index[1],self.node_threshold[1],self.node_entropy[1])[3])
    self.node_entropy.append(self.information_gain(self.node_feature[1],self.node_data_index[1],self.node_threshold[1],self.node_entropy[1])[4])
    self.node_data.append(self.information_gain(self.node_feature[1],self.node_data_index[1],self.node_threshold[1],self.node_entropy[1])[0])
    self.node_data.append(self.information_gain(self.node_feature[1],self.node_data_index[1],self.node_threshold[1],self.node_entropy[1])[1])
    for node_index in range [4:2**int(layer_num+1)]:
      self.node_entropy.append(0)
      self.node_data_index.append(0)
    for node_index in range [2:2**int(layer_num+1)]:
      self.node_feature.append(0)
      self.node_threshold.append(0)
    for node_index in range [2:2**int(layer_num)]:
      self.node_feature[node_index]=self.feature_threshold_decision(self.node_data_index[parent(node_index)])[0]
      self.node_treshold[node_index]=self.threshold_decision(self.node_feature[node_index],self.node_data_index[node_index],self.node_entropy[node_index])
      self.node_data_index[2*node_index]=self.information_gain(self.node_feature[node_index],self.node_data_index[node_index],self.node_threshold[node_index],self.node_entropy[node_index])[0]
      self.node_data_index[2*node_index+1]=self.information_gain(self.node_feature[node_index],self.node_data_index[node_index],self.node_threshold[node_index],self.node_entropy[node_index])[1]
      self.node_entropy[2*node_index]=self.information_gain(self.node_feature[node_index],self.node_data_index[node_index],self.node_threshold[node_index],self.node_entropy[node_index])[3]
      self.node_entropy[2*node_index+1]=self.information_gain(self.node_feature[node_index],self.node_data_index[node_index],self.node_threshold[node_index],self.node_entropy[node_index])[4]
   
  def node_walk(test_index,node_index):
    if(self.test[test_index][self.node_feature[node_index]]>self.node_threshold[self.node_feature[node_index]]*0.1*self.feature_max[self.node_fearture[node_index]]):
      return(0)
    else:
      return(1)

  def decide_class(test_data_index,layer_num):
    path = [0]*(layer_num+1)
    path[0] = 1
    for path_index in range [0:layer_num]:
      if(node_walk(test_data_index,path[path_index])):
        path[path_index]=2*path_index+1
      else:
        path[path_index]=2*path_index
    if((path[layer_num+1]%2)==0):
      return (1)
    else:
      return (0)
       

class random_forest:

  DT = [] 
  test = []
  train = []
  layer_num = 10 
 
  def read_train(self):
    with open('spam_train.csv') as data_file:
      for line in data_file:
        self.train.append(line.strip().split(','))
    data_file.close()
    for i in (len(train)):
      for j in (len(train[0])):
        self.train[i][j]=int(self.train[i][j])
    for i in range (4001):
      for j in range (57):
        self.feature[j][i]=self.train[i][j+1]
      self.classify[i] = self.train[i][58]
    for i in range (57):
      for k in range(4001):
        if(self.feature[i][k]>self.feature_max[i]):
          self.feature_max[i]=self.feature[i][k]
    
  def read_test(self):
    with open('spam_test.csv') as data_file:
      for line in data_file:
        self.test.append(line.strip().split(','))
    data_file.close()
    for i in range (len(self.test)):
      for j in range (len(self.test[0])):
        self.test[i][j]=int(self.test[i][j])

  def _init_(self,layer_num):
    read_train()
    read_test()
    for feature_index in range(57):
      self.DT.append(decision_tree(feature_index,layer_num))
     
  def majority_vote(self,test_data_index,layer_num):
    classify_0 = 0
    classify_1 = 0
    for tree_index in range (57):
      if(self.DT[tree_index].decide_class(test_data_index,layer_num)):
        classify_1 = classify_1 + 1 
      else:
        classify_0 = classify_0 + 1
    if(classify_1 > classify_0):
      return(1)
    else:
      return(0) 
  
  def print_classify(self,layer_num):
    f = open('label.csv', 'w+')
    f.write('index,label\n')
    for test_data_index in range (600):
      f.write(str(test_data_index+1))
      f.write(',')
      f.write(str(self.majority_vote(test_data_index,layer_num)))
      f.write('\n')
        

RF = random_forest(10)
RF.print_classify(10)

      
    
    
  
      












    
