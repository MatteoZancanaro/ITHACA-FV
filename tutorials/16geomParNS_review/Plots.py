import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc

x = np.linspace(0,1,100)

def f1(x):
   return (x**0.5 * (1 - x)) / (np.exp(15 * x))

def f2(x):
   return (np.sin(np.pi*(x**(0.25))))**3

def f3(x):
   return (np.sin(np.pi*(x**(0.757))))**3

def f4(x):
   return (np.sin(np.pi*(x**(1.357))))**3

def f5(x):
   return (x**0.5 * (1 - x)) / (np.exp(10 * x))

fig = plt.figure()
plt.grid(True,color='grey',linestyle='-')
plt.plot(x,f1(x),label='f1', linewidth=2)
plt.plot(x,f2(x),label='f2', linewidth=2)
plt.plot(x,f3(x),label='f3', linewidth=2)
plt.plot(x,f4(x),label='f4', linewidth=2)
plt.plot(x,f5(x),label='f5', linewidth=2)
plt.legend()
plt.ylabel(r'$f(x) \cdot C$',fontsize=20)
plt.xlabel(r"$x / C$",fontsize=20)
fig.savefig("BumpFunctions.pdf", bbox_inches='tight')
plt.show()








