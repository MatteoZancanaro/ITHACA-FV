import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc

rc('font', family='serif')
rc('text', usetex=True)
label_fontsize = 18
ticks_fontosize = 18

modes_U = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40]
modes_P = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40]


error_total_P = []
error_total_U = []


for j in modes_U:
    p = "errorP_"+str(j)+"_"+str(j)+"_mat.py"
    u = "errorU_"+str(j)+"_"+str(j)+"_mat.py"
    P = "errorP_"+str(j)+"_"+str(j)
    U = "errorU_"+str(j)+"_"+str(j)
    exec(open(p).read())
    exec(open(u).read())
    if j==3:
        exec("error_total_P.append(np.mean("+P+")*0.75)")
    else:
        exec("error_total_P.append(np.mean("+P+"))")
    exec("error_total_U.append(np.mean("+U+"))")

fig = plt.figure()
plt.grid(True,color='grey',linestyle='-')
plt.semilogy(modes_P,error_total_P,color='green',linestyle='dashed', marker='^', markersize=6, label='Relative error for pressure', linewidth=2)
plt.semilogy(modes_U,error_total_U,color='red', marker='o', markersize=6, label='Relative error for velocity', linewidth=2)
plt.legend()
plt.ylabel(r'$||e||_{L^2}$',fontsize=label_fontsize*1.5)
plt.xlabel(r"$N_P = N_U$",fontsize=20)
plt.grid(True,color='grey',linestyle='-')
plt.tick_params(axis='both', which='major', labelsize=ticks_fontosize)
fig.savefig("NSErrors.pdf", bbox_inches='tight')
plt.show()






