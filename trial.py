import numpy as np
from numpy.matlib import repmat
data=np.arange(100)
data=data.reshape((10,10))


xdata = [y*np.ones((3,3)) for x in data for y in x]
xdata=np.array(xdata).reshape((30,30))

print(np.shape(xdata))
print(xdata)
