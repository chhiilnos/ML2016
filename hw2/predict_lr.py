import csv
import sys
import numpy

w = [0]*57
test = []

with open(sys.argv[2]) as data_file:
  for line in data_file:
    test.append(line.strip().split(','))
data_file.close()

with open(sys.argv[1]) as data_file:
  for line in data_file:
    w[i]=float(line)
data_file.close()

def f_wb(self,x):
  z = 0.
  for j in range (57):
    z=z+w[j]*float(x[j])
  f_wb=1/(1+numpy.exp(-z))
  return(f_wb)  
      
f = open(output_file, 'w+')
f.write('id,label\n')
for k in range(600):
  f.write(str(k+1))
  f.write(',')
  if(f_wb(test[k][1:])>0.5):
    f.write('1\n')
  else:
    f.write('0\n')
f.close()
  
