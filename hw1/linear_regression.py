import numpy
import sys
train = []

NO2 = []
O3 = []
PM10 = []
PM = []
RAINFALL = []

no2 = []
o3 = []
pm10 = []
pm = []
rainfall = []
rate = [0.]*5
rd = 1000
###read in data from train
with open('train.csv') as data_file:
  for line in data_file:
     train.append(line.strip().split(','))

for i in range (0,4321):
  if (train[i][2]=='NO2'):
    NO2.append(train[i][3:27])
  if (train[i][2]=='O3'):
    O3.append(train[i][3:27])
  if (train[i][2]=='PM10'):
    PM10.append(train[i][3:27])
  if (train[i][2]=='PM2.5'):
    PM.append(train[i][3:27])
  if (train[i][2]=='RAINFALL'):
    RAINFALL.append(train[i][3:27])


print('NO2')
for i in range (0,len(NO2)):
  for j in range (0,24):
    no2.append(float(NO2[i][j]))
print('O3')
for i in range (0,len(O3)):
  for j in range (0,24):
    o3.append(float(O3[i][j]))
print('PM10')
for i in range (0,len(PM10)):
  for j in range (0,24):
    pm10.append(float(PM10[i][j]))
print('PM')
for i in range (0,len(PM)):
  for j in range (0,24):
    pm.append(float(PM[i][j]))
print('RAINFALL')
for i in range (0,len(RAINFALL)):
  for j in range (0,24):
    if(RAINFALL[i][j]=='NR'):
      rainfall.append(0.)
    else:
      rainfall.append(float(RAINFALL[i][j]))

del train[:]

del NO2[:]
del O3[:]
del PM10[:]
del PM[:]
del RAINFALL[:]
data_file.close()

###read in data from test_X
test_X = []

NO2 = []
O3 = []
PM10 = []
PM = []
RAINFALL = []

with open('test_X.csv') as data_file:
  for line in data_file:
     test_X.append(line.strip().split(','))

for i in range (0,4320):
	if (test_X[i][1]=='NO2'):
		NO2.append(test_X[i][2:11])
	if (test_X[i][1]=='O3'):
		O3.append(test_X[i][2:11])
	if (test_X[i][1]=='PM10'):
		PM10.append(test_X[i][2:11])
	if (test_X[i][1]=='PM2.5'):
		PM.append(test_X[i][2:11])
	if (test_X[i][1]=='RAINFALL'):
		RAINFALL.append(test_X[i][2:11])
	
	
print('NO2')
for i in range (0,len(NO2)):
	for j in range (0,9):
		NO2[i][j]=float(NO2[i][j])
print('O3')
for i in range (0,len(O3)):
	for j in range (0,9):
		O3[i][j]=float(O3[i][j])
print('PM10')
for i in range (0,len(PM10)):
	for j in range (0,9):
		PM10[i][j]=float(PM10[i][j])
print('PM')
for i in range (0,len(PM)):
	for j in range (0,9):
		PM[i][j]=float(PM[i][j])
print('RAINFALL')
for i in range (0,len(RAINFALL)):
	for j in range (0,9):
		if(RAINFALL[i][j]=='NR'):
			RAINFALL[i][j]=(0.)
		else:
			RAINFALL[i][j]=(float(RAINFALL[i][j]))


del test_X[:]

data_file.close()

###define some functions
def regulization():
  max=[no2[0],o3[0],pm10[0],pm[0],rainfall[0]]
  min=[no2[0],o3[0],pm10[0],pm[0],rainfall[0]]
  for i in range(5760):
    if(no2[i]<min[0]):
      min[0]=no2[i]
    if(no2[i]>max[0]):
      max[0]=no2[i]
    if(o3[i]<min[1]):
      min[1]=o3[i]
    if(o3[i]>max[1]):
      max[1]=o3[i]
    if(pm10[i]<min[2]):
      min[2]=pm10[i]
    if(pm10[i]>max[2]):
      max[2]=pm10[i]
    if(pm[i]<min[3]):
      min[3]=pm[i]
    if(pm[i]>max[3]):
      max[3]=pm[i]
    if(rainfall[i]<min[4]):
      min[4]=rainfall[i]
    if(rainfall[i]>max[4]):
      max[4]=rainfall[i]
  for i in range(5):
    rate[i]=max[i]-min[i]
  for i in range(5760):
    no2[i]=no2[i]/(max[0]-min[0])
    o3[i]=o3[i]/(max[1]-min[1])
    pm10[i]=pm10[i]/(max[2]-min[2])
    pm[i]=pm[i]/(max[3]-min[3])
    rainfall[i]=rainfall[i]/(max[4]-min[4])
    
def func(a,b):
  F=0.
  for m in range(12):
    for n in range(471):
      f=0.
      for i in range(1):
        f=f-a[i]*no2[m*480+n+8-i]
        f=f-a[i+20]*no2[m*480+n+8-i]*no2[m*480+n+8-i]
      for i in range(3):
        f=f-a[i+1]*o3[m*480+n+8-i]
        f=f-a[i+21]*o3[m*480+n+8-i]*o3[m*480+n+8-i]
      for i in range(5):
        f=f-a[i+4]*pm10[m*480+n+8-i]
        f=f-a[i+24]*pm10[m*480+n+8-i]*pm10[m*480+n+8-i]
      for i in range(9):
        f=f-a[i+9]*pm[m*480+n+8-i]
        f=f-a[i+29]*pm[m*480+n+8-i]*pm[m*480+n+8-i]
      for i in range(2):
        f=f-a[i+18]*rainfall[m*480+n+8-i]
        f=f-a[i+38]*rainfall[m*480+n+8-i]*rainfall[m*480+n+8-i]
      f=f+pm[m*480+n+9]-b
      F=F+f*f
  return(F)
  

def func_grad(a,b):
  ga = [0.]*40
  gb = 0.
  for m in range(12):
      for n in range(471):
        f=0.
        for i in range(1):
          f=f-a[i]*no2[m*480+n+8-i]
          f=f-a[i+20]*no2[m*480+n+8-i]*no2[m*480+n+8-i]
        for i in range(3):
          f=f-a[i+1]*o3[m*480+n+8-i]
          f=f-a[i+21]*o3[m*480+n+8-i]*o3[m*480+n+8-i]
        for i in range(5):
          f=f-a[i+4]*pm10[m*480+n+8-i]
          f=f-a[i+24]*pm10[m*480+n+8-i]*pm10[m*480+n+8-i]
        for i in range(9):
          f=f-a[i+9]*pm[m*480+n+8-i]
          f=f-a[i+29]*pm[m*480+n+8-i]*pm[m*480+n+8-i]
        for i in range(2):
          f=f-a[i+18]*rainfall[m*480+n+8-i]
          f=f-a[i+38]*rainfall[m*480+n+8-i]*rainfall[m*480+n+8-i]
        f=f+pm[m*480+n+9]-b
        for i in range(1):
          ga[i]=ga[i]-2*f*no2[m*480+n+8-i]
          ga[i+20]=ga[i+20]-2*f*no2[m*480+n+8-i]*no2[m*480+n+8-i]
        for i in range(3):
          ga[i+1]=ga[i+1]-2*f*o3[m*480+n+8-i]
          ga[i+21]=ga[i+21]-2*f*o3[m*480+n+8-i]*o3[m*480+n+8-i]
        for i in range(5):
          ga[i+4]=ga[i+4]-2*f*pm10[m*480+n+8-i]
          ga[i+24]=ga[i+24]-2*f*pm10[m*480+n+8-i]*pm10[m*480+n+8-i]
        for i in range(9):
          ga[i+9]=ga[i+9]-2*f*pm[m*480+n+8-i]
          ga[i+29]=ga[i+29]-2*f*pm[m*480+n+8-i]*pm[m*480+n+8-i]
        for i in range(2):
          ga[i+18]=ga[i+18]-2*f*rainfall[m*480+n+8-i]
          ga[i+38]=ga[i+38]-2*f*rainfall[m*480+n+8-i]*rainfall[m*480+n+8-i]
        gb=gb-2*f
  return(ga,gb)


def run_adagrad(x,y,eta):
  Gx = [0.]*40
  Gy = 0.
  cost = [0.]*rd
  for k in range(rd):
    print('adagrad:{}'.format(k))
    (gx,gy)=func_grad(x,y)
    for i in range(20):
      Gx[i]=Gx[i]+gx[i]*gx[i]
    Gy=Gy+gy*gy
    for i in range(20):
      x[i]=x[i]-eta*(1.0/(Gx[i]**0.5))*gx[i]
      y=y-eta*(1.0/(Gy**0.5))*gy
    cost[k]=(func(x,y)/(12*471))**0.5
  
  return (x,y,cost)

def run_adagrad_reg(x,y,eta):
  Gx = [0.]*40
  Gy = 0.
  cost = [0.]*rd
  for k in range(rd):
    print('adagrad:{}'.format(k))
    (gx,gy)=func_grad(x,y)
    for i in range(20):
      Gx[i]=Gx[i]+gx[i]*gx[i]
    Gy=Gy+gy*gy
    for i in range(20):
      x[i]=x[i]-eta*(1.0/(Gx[i]**0.5))*gx[i]
      y=y-eta*(1.0/(Gy**0.5))*gy
    cost[k]=((func(x,y)/(12*471))**0.5)*rate[3]
  
  return (x,y,cost)

###no reg and run adagrad test
u = [0.]*40
v = 0.

(t,tc,C)=run_adagrad(u,v,1000)



###calculate the value

value=[0.]*240
for i in range (0,240):
  for k in range(1):
    value[i]=value[i]+NO2[i][8-k]*t[k]
    value[i]=value[i]+NO2[i][8-k]*NO2[i][8-k]*t[k+20]
  for k in range(3):
    value[i]=value[i]+O3[i][8-k]*t[k+1]
    value[i]=value[i]+O3[i][8-k]*O3[i][8-k]*t[k+21]
  for k in range(5):
    value[i]=value[i]+PM10[i][8-k]*t[k+4]
    value[i]=value[i]+PM10[i][8-k]*PM10[i][8-k]*t[k+24]
  for k in range(9):
    value[i]=value[i]+PM[i][8-k]*t[k+9]
    value[i]=value[i]+PM[i][8-k]*PM[i][8-k]*t[k+29]
  for k in range(2):
    value[i]=value[i]+RAINFALL[i][8-k]*t[k+18]
    value[i]=value[i]+RAINFALL[i][8-k]*RAINFALL[i][8-k]*t[k+38]
  value[i]=value[i]+tc


###writing the result to linear_regression.csv
f = open('linear_regression.csv', 'w+')
f.write('id,value\n')
for i in range(240):
	f.write('id_{},{}\n'.format(i,value[i]))
f.close()


###write cost function without regulation,eta =1000

f = open('cost.csv', 'w+')
for k in range(rd):
	f.write('{}\n'.format(C[k]))
f.close()


###write cost function eta=10
for i in range(40):
  u[i]=0.
v = 0.

regulization()
(t,tc,C)=run_adagrad(u,v,10)

f = open('eta_10.csv', 'w+')
for k in range(rd):
	f.write('{}\n'.format(C[k]))
f.close()

###print cost function eta=0.1
for i in range(40):
  u[i]=0.
v = 0.

regulization()
(t,tc,C)=run_adagrad(u,v,0.1)

f = open('eta_0.1.csv', 'w+')
for k in range(rd):
	f.write('{}\n'.format(C[k]))
f.close()
###write cost function with regulation

for i in range(40):
  u[i]=0.
v = 0.

regulization()
(t,tc,C)=run_adagrad_reg(u,v,1000)
f = open('cost_reg.csv', 'w+')
for k in range(rd):
	f.write('{}\n'.format(C[k]))
f.close()


del no2[:]
del o3[:]
del pm10[:]
del pm[:]
del rainfall[:]




