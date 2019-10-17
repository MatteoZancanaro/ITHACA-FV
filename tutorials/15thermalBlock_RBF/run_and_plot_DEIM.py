import numpy as np
import os
import files 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

modes_T = [25,25,25,25,25,25,25]
modes_DEIM = [5,10,15,20,25,30,40]

for k,j in zip(modes_T, modes_DEIM):
     files.sed_variable("N_modes_T","./system/ITHACAdict",k)
     files.sed_variable("N_modes_DEIM_A","./system/ITHACAdict",j)
     files.sed_variable("N_modes_DEIM_B","./system/ITHACAdict",j)
     os.system("10thermalBlock_RBF")

error_total=[]
error=[]

for k,j in zip(modes_T, modes_DEIM):
    s = "error_"+str(k)+"_"+str(j)+"_"+str(j)+"_mat.py"
    m = "error_"+str(k)+"_"+str(j)+"_"+str(j)
    exec(open(s).read())
    exec("error_total.append("+m+")")

for j in range(0,len(modes_DEIM)):
    error.append(np.mean(error_total[j]))

print(error)

pp = PdfPages('./multipage.pdf')

plt.semilogy(modes_DEIM,error,':o', label='Relative error for ROM')
# plt.semilogy(PRO[:,0],PRO[:,1],'k--v', label='Relative error for L2 proj.')
# plt.xlim(5,50)
plt.xlabel("$N$ of modes")
plt.ylabel("L2 Rel. Error.")

# plt.legend(bbox_to_anchor=(.5,  .95), loc=2, borderaxespad=0.) 
plt.grid(True)
plt.show()
plt.savefig(pp, format='pdf')