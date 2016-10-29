import numpy
import sys

class two_set_classify:
 
  train = []
  test = []
  w = [0.]*57
  b = 0.
  feature_name = []
  feature = [[0 for x in range(4001)] for y in range(57)]
  classify = [0.]*4001

  def read_train(self,file_name):
    with open(file_name) as data_file:
      for line in data_file:
        self.train.append(line.strip().split(','))
    data_file.close()
    for i in range (4001):
      for j in range (57):
        self.feature[j][i]=self.train[i][j+1]
      self.classify[i] = self.train[i][58]
    

  def f_wb(self,x):
    z = 0.
    for j in range (57):
      z=z+self.w[j]*float(x[j])
    z=z+self.b
    f_wb=1/(1+numpy.exp(-z))
    return(f_wb)  
      
  def logistic_regression(self,n,eta):
    for i in range(n):
      S=[1.]*57
      for j in range (57):
        s=0.
        for k in range(4001):
          s=s+(float(self.classify[k])-self.f_wb(self.train[k][1:58]))*float(self.train[k][j+1])
          S[j]=S[j]+s*s
        self.w[j]=self.w[j]+eta*(1.0/(S[j]**0.5))*s

  def write_w(self,file_name):
    f = open(file_name, 'w+')
    f.write('id,label\n')
    for i in range(57):
      f.write(str(w[i]))
      f.write('\n')
  
  def output(self,input_file,output_file):
    with open(input_file) as data_file:
      for line in data_file:
        self.w[i]=float(line)
    data_file.close()
    f = open(output_file, 'w+')
    f.write('id,label\n')
    for k in range(600):
      f.write(str(k+1))
      f.write(',')
      if(self.f_wb(self.test[k][1:58])>0.5):
        f.write('1\n')
      else:
        f.write('0\n')
    f.close()

model = two_set_classify()
model.read_train(sys.argv[1])
model.logistic_regression(110, 5)
model.write_w(sys.argv[2])



