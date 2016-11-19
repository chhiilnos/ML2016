from keras.models import load_model
import csv
import sys
import pickle
import numpy as np

model=load_model(sys.argv[2])

test = pickle.load(open(sys.argv[1]+'test.p','rb'))
X_test = np.array(test['data'])
X_test = X_test.reshape((10000,3,32,32))

result = model.predict(X_test)

f=open(sys.argv[3],'w')
f.write('ID,class\n')
for i in range (10000):
  f.write('{},{}\n'.format(i,np.argmax(result[i])))
f.close()


