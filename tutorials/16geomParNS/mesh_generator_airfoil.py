from naca import *
import numpy as np
import matplotlib.pyplot as plt

# see https://www.ehsanmadadi.com/naca-4-digit-airfoil-mesh-using-blockmesh-with-m4-macro/ for the blocks structure

def spl(c,c1,c2):
	ok = []
	for i in c:
		if i[0]<=max(c2[0],c1[0]) and i[1]<=max(c2[1],c1[1]) and i[0]>=min(c2[0],c1[0]) and i[1]>=min(c2[1],c1[1]):
			ok.append(i)
	ok = np.asarray(ok)
	ok = ok[ok[:,0].argsort()]
	return ok

###################### Input Parameters Mesh ###########################
			
# WING Parameters
np_naca = 100			# Number of points along the naca
angle = 0 			# Angle of attack of the naca
naca_type = '4412'		# Naca type (pos. choiches 2414 , 0009 , 6409)
scale = float(1)		# Scale the naca (leave 1)

#Height of mesh in y direction
L1 = 8.0

#Length of downstream
L2 = 16.0

#Foil depth in z direction perpendicular to x-y surface
L3 = 0.2

#Number of cells and points at each direction and element
#Number of cells in y direction
Nl1 = 100

#Number of cells in downstream
Nl2 = 100

#Number of cells in z direction
Nl3 = 1

#Number of meshes on the front part of airfoil edges p8-p9 and p8-p10b TODO calc it based on number of elements on the spline
Nl4 = 50

#Number of meshes on the back part of airfoil edges p9-p11 and p10-p11 TODO calc it based on number of elements on the spline
Nl5 = 71

#Number of interpolation points along the airfoil for defining the splines
Naf = 50

#Cell expansion ratios

#Expansion ratio in y direction
E1 = 50

#Expansion ratio in downstream side
E2 = 100

#Expansion ratio in inlet
E3 = 2

#Expansion ratio in inlet2
E5 = 1

#Expansion ratio in y
E4 = 1/E1

# Base z
Zb = 0

# Depth of airfoil
zd = 0.2

#Front z
Zt = Zb + zd


###### Design the naca profile and center respect to the domain ########
x,z = naca(naca_type,np_naca,False,True)
x = np.array(x)
z = np.array(z)
np_wing = x.size
xr = x*np.cos(angle*pi/180) - z*np.sin(angle*pi/180)
zr = x*np.sin(angle*pi/180) + z*np.cos(angle*pi/180)
x=xr
z=zr
x=x-x.min()
z=z-z.min()
z = z - (z.max() - z.min())/2
x = x - (x.max() - x.min())/2
x=x*scale
z=z*scale

c = np.zeros((np_naca*2+1,3))
c[:,0] = x
c[:,1] = z
c[:,2] = z*0

c2 = c.copy()

c2[:,2] = c2[:,2]+L3

#Wings
maxVals = np.argmax(c,axis=0);
minVals = np.argmin(c,axis=0);

P08 = [c[minVals[0],0],c[minVals[0],1],c[minVals[0],2]]
P11 = [c[maxVals[0],0],c[maxVals[0],1],c[maxVals[0],2]]
P09 = [c[maxVals[1],0],c[maxVals[1],1],c[maxVals[1],2]]
P10 = [c[minVals[1],0],c[minVals[1],1],c[minVals[1],2]]
P07 = [P09[0],L1,0,0]
P01 = [P10[0],-L1,0,0]
P00 = [-L1 + P08[0], P08[1],0]
P04 = [L2 + P11[0], P11[1],0]
P02 = [P11[0],P01[1],0]
P06 = [P11[0],P07[1],0]
P05 = [P04[0],P06[1],0]
P03 = [P04[0],P02[1],0]
P12 =  [P00[0],P00[1],L3]
P13 =  [P01[0],P01[1],L3]
P14 =  [P02[0],P02[1],L3]
P15 =  [P03[0],P03[1],L3]
P16 =  [P04[0],P04[1],L3]
P17 =  [P05[0],P05[1],L3]
P18 =  [P06[0],P06[1],L3]
P19 =  [P07[0],P07[1],L3]
P20 =  [P08[0],P08[1],L3]
P21 =  [P09[0],P09[1],L3]
P22 =  [P10[0],P10[1],L3]
P23 =  [P11[0],P11[1],L3]

PAU = [-L1*np.cos(45*np.pi/180)+P08[0],L1*np.sin(45*np.pi/180),0]
PAB = [-L1*np.cos(45*np.pi/180)+P08[0],-L1*np.sin(45*np.pi/180),0]

PAU2 = [PAU[0],PAU[1],L3]
PAB2 = [PAB[0],PAB[1],L3]

p0810 = spl(c,P08,P10)
p1011 = spl(c,P10,P11)
p0809 = spl(c,P08,P09)
p0911 = spl(c,P09,P11)
p2022 = spl(c2,P20,P22)
p2223 = spl(c2,P22,P23)
p2021 = spl(c2,P20,P21)
p2123 = spl(c2,P21,P23)

vertices = [P00,P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12,P13,P14,P15,P16,P17,P18,P19,P20,P21,P22,P23]

print("/*--------------------------------*- C++ -*----------------------------------*\\")
print("| =========                 |                                                 |")
print("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |")
print("|  \\\\    /   O peration     |                                                 |")
print("|   \\\\  /    A nd           | Copyright (C) 2018 Ehsan Madadi-Kandjani        |")
print("|    \\\\/     M anipulation  |                                                 |")
print("\\*---------------------------------------------------------------------------*/")
print("FoamFile")
print("{")
print("    version     2.0;")
print("    format    ascii;")
print("    class       dictionary;")
print("    object      blockMeshDict;")
print("}")
print("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //")

print("convertToMeters 1.0;")

## Print Vertices
print("vertices\n(")
for i in vertices:
	print("("+str(i[0])+" "+str(i[1])+" "+str(i[2])+")")
print(");")

## Print Vertices
print("blocks\n(")
print("// Block 0")
print("hex (0 1 10 8 12 13 22 20)")
print("square")
print("("+str(Nl4)+" "+str(Nl1)+" 1)")
print("simpleGrading")
print("("+str(E3)+" "+str(E4)+" 1)\n")

print("// Block 1")
print("hex (1 2 11 10 13 14 23 22)")
print("square")
print("("+str(Nl5)+" "+str(Nl1)+" 1)")
print("simpleGrading")
print("("+str(E5)+" "+str(E4)+" 1)\n")

print("// Block 2")
print("hex (2 3 4 11 14 15 16 23)")
print("square")
print("("+str(Nl2)+" "+str(Nl1)+" 1)")
print("simpleGrading")
print("("+str(E2)+" "+str(E4)+" 1)\n")

print("// Block 3")
print("hex (11 4 5 6 23 16 17 18)")
print("square")
print("("+str(Nl2)+" "+str(Nl1)+" 1)")
print("simpleGrading")
print("("+str(E2)+" "+str(E1)+" 1)\n")

print("// Block 4")
print("hex (9 11 6 7 21 23 18 19)")
print("square")
print("("+str(Nl5)+" "+str(Nl1)+" 1)")
print("simpleGrading")
print("("+str(E5)+" "+str(E1)+" 1)\n")

print("// Block 5")
print("hex (8 9 7 0 20 21 19 12)")
print("square")
print("("+str(Nl4)+" "+str(Nl1)+" 1)")
print("simpleGrading")
print("("+str(E3)+" "+str(E1)+" 1)\n")

print(");")

## Print Edges
print("edges\n(")
print("arc 0 7 ("+str(PAU[0])+" "+str(PAU[1])+" "+str(PAU[2])+")")
print("arc 0 1 ("+str(PAB[0])+" "+str(PAB[1])+" "+str(PAB[2])+")")
print("arc 12 19 ("+str(PAU2[0])+" "+str(PAU2[1])+" "+str(PAU2[2])+")")
print("arc 12 13 ("+str(PAB2[0])+" "+str(PAB2[1])+" "+str(PAB2[2])+")")

print("spline 8 10")
print("(")
for k in p0810:
	print("("+str(k[0])+" "+str(k[1])+" "+str(k[2])+")")
print(")")

print("spline 10 11")
print("(")
for k in p1011:
	print("("+str(k[0])+" "+str(k[1])+" "+str(k[2])+")")
print(")")

print("spline 8 9")
print("(")
for k in p0809:
	print("("+str(k[0])+" "+str(k[1])+" "+str(k[2])+")")
print(")")

print("spline 9 11")
print("(")
for k in p0911:
	print("("+str(k[0])+" "+str(k[1])+" "+str(k[2])+")")
print(")")

print("BSpline 20 22")
print("(")
for k in p2022:
	print("("+str(k[0])+" "+str(k[1])+" "+str(k[2])+")")
print(")")

print("BSpline 22 23")
print("(")
for k in p2223:
	print("("+str(k[0])+" "+str(k[1])+" "+str(k[2])+")")
print(")")

print("BSpline 20 21")
print("(")
for k in p2021:
	print("("+str(k[0])+" "+str(k[1])+" "+str(k[2])+")")
print(")")

print("BSpline 21 23")
print("(")
for k in p2123:
	print("("+str(k[0])+" "+str(k[1])+" "+str(k[2])+")")
print(")")
print(");")

print("boundary\n(")
print("    ")
print("    inlet\n{")
print("        type patch;\nfaces\n(")
print("        (7 0 12 19)")
print("        (0 1 13 12)")
print("    );\n}")
print("    ")
print("    outlet\n{")
print("        type patch;\nfaces\n(")
print("        (4 5 17 16)")
print("        (3 4 16 15)")
print("        (6 7 19 18)")
print("        (5 6 18 17)")
print("        (2 3 15 14)")
print("        (1 2 14 13)")
print("    );\n}")
print("    ")
print("    walls\n{")
print("        type wall;\nfaces\n(")
print("        (8 10 22 20)")
print("        (10 11 23 22)")
print("        (8 9 21 20)")
print("        (9 11 23 21)")
print("    );\n}")
print("    ")
print("    frontAndBack\n{")
print("        type empty;\nfaces\n(")
print("        (8 0 1 10)")
print("        (11 10 1 2)")
print("        (4 11 2 3)")
print("        (7 0 8 9)")
print("        (6 7 9 11)")
print("        (5 6 11 4)")
print("        (20 12 13 22)")
print("        (23 22 13 14)")
print("        (16 23 14 15)")
print("        (19 12 20 21)")
print("        (18 19 21 23)")
print("        (17 18 23 16)")
print("    );\n}")
print(");")
print("")
print("mergePatchPairs")
print("(")
print(");")
