import numpy as np
from tqdm import tqdm
import os

os.makedirs("data", exist_ok=True)

B = np.linspace(-10,10,256)

def lorentzian_derivative(B,B0,gamma,A):
    return -2*A*(B-B0)/(gamma**2*(1+((B-B0)/gamma)**2)**2)

def generate_radical():

    B0=np.random.uniform(-5,5)
    gamma=np.random.uniform(0.3,1.5)
    A=np.random.uniform(0.5,1.2)

    return lorentzian_derivative(B,B0,gamma,A)

def generate_sample():

    r1=generate_radical()
    r2=generate_radical()

    composite=r1+r2

    noise=np.random.normal(0,0.02,len(B))

    composite=composite+noise
    scale = np.max(np.abs(composite)) + 1e-8
    composite = composite / scale
    r1 = r1 / scale
    r2 = r2 / scale

    return composite,r1,r2

X=[]
Y=[]

for i in tqdm(range(10000)):

    c,r1,r2=generate_sample()

    X.append(c)
    Y.append([r1,r2])

X=np.array(X)
Y=np.array(Y)

np.save("data/X.npy",X)
np.save("data/Y.npy",Y)