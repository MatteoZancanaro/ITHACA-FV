import numpy as np
import os
import files 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

cumEigsU = np.loadtxt("ITHACAoutput/POD/CumEigenvalues_p",skiprows=2)
cumEigsP = np.loadtxt("ITHACAoutput/POD/CumEigenvalues_U",skiprows=2)
eigsU = np.loadtxt("ITHACAoutput/POD/Eigenvalues_p",skiprows=2)
eigsP = np.loadtxt("ITHACAoutput/POD/Eigenvalues_U",skiprows=2)
eigsNum = eigsU.size
xEig = np.linspace(1,eigsNum,eigsNum)

old = [0.007784290, 0.0077489, 0.00775245, 0.00779249, 0.00787784, 0.00800721, 0.00817655, 0.00837321, 0.00855936, 0.00878705, 0.00905042, 0.00934797, 0.00966687, 0.00999854, 0.0103237, 0.0106045, 0.0108182, 0.0110847, 0.0113754, 0.0116899, 0.0120157, 0.012347, 0.0126904, 0.013046, 0.0133994, 0.0137633, 0.0141359, 0.0145093, 0.0148888, 0.0152731, 0.0156614, 0.0160462, 0.0164332, 0.0168219, 0.0172118, 0.0176023, 0.0179867, 0.0183713, 0.0187313, 0.0191465, 0.019531, 0.0199156, 0.0203005, 0.0206861, 0.0210726, 0.0214604, 0.0218497, 0.022247, 0.0226408, 0.0230374 ]

new = [0.00236024, 0.00225459, 0.00216186, 0.00216153, 0.00224879, 0.00230806, 0.00229361, 0.00220612, 0.00208194, 0.00202043, 0.00197937, 0.00194663, 0.00191979, 0.0018987, 0.00188029, 0.0018553, 0.00182111, 0.0017934, 0.00176769, 0.00174404, 0.00172234, 0.00170243, 0.0016845, 0.00166843, 0.00165359, 0.00164021, 0.00162797, 0.00161616, 0.00160473, 0.00159334, 0.00158186, 0.00156994, 0.0015577, 0.00154498, 0.00153186, 0.00151845, 0.00150493, 0.00149138, 0.00147847, 0.00146536, 0.00145341, 0.00144241, 0.00143276, 0.00142471, 0.00141857, 0.00141471, 0.00141358, 0.00141532, 0.00142091, 0.00143077]

solNum = len(new)

xErr = np.linspace(1,solNum,solNum)

eigsData = {'xEig': xEig, 'eigsU': eigsU, 'eigsP': eigsP, 'cumEigsU': cumEigsU, 'cumEigsP': cumEigsP, 'xErr': xErr, 'oldErr': old, 'newErr': new}
outputEigs = pd.DataFrame(eigsData)
#print(outputEigs) 
outputEigs.to_csv("incLamBackData.dat",sep=' ',index=False)

exit()





