import numpy as np

import math

def fnet(p):#continous neuron

    return 1/(1+math.exp(-lamb*p))

def flin(l):#linear neuron

    return l

def differential(s):#continous neuron

    return s*(1-s)

eta=1

fdnetk=1#linear neuron

v = np.matrix('1 -2 3; 2 0 -1')

w= np.matrix('1 0 2')

z= np.matrix('1;3;-1')

d=1

lamb=1

netj=np.dot(v,z)

Y=np.reshape(np.matrix(np.array([fnet(netj.item(0)),fnet(netj.item(1)),-1])),(3,1))

netk=np.dot(w,Y)

ok=flin(netk)

p1=differential(Y.item(0))

p2=differential(Y.item(1))

fdnetj=np.reshape(np.matrix(np.array([p1,p2,0])),(3,1))

E=0.5*(d-ok)**2

deltaok=fdnetk*(d-ok)

deltaokwkj=deltaok*w

deltayj=np.dot(fdnetj,deltaokwkj)

deltaw= eta*deltaok.item(0) *Y

deltav=eta*np.dot(deltayj,z)

Wnew=w.T + deltaw

v1= np.reshape(np.matrix(np.array([v.item(0),v.item(3)])),(2,1))

v2= np.reshape(np.matrix(np.array([v.item(1),v.item(4)])),(2,1))

v3= np.reshape(np.matrix(np.array([v.item(2),v.item(5)])),(2,1))

vx=np.reshape(np.delete(deltav,2),(2,1))

v1=vx+v1

v2=vx+v2

v3=vx+v3

Vnew=np.c_[v1, v2,v3]



print "Wnew ",Wnew

print "Vnew ",Vnew
