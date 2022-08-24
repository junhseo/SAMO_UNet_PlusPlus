from odbAccess import *
from numpy import array, concatenate, zeros, genfromtxt, savetxt, sort, unique, where
import sys

odb          = openOdb(path='PLATE_STATIC.odb')
lastFrame    = odb.steps['STATIC'].frames[-1]

#############################################################################################
# GLOBAL NODAL DATA EXPORT
#############################################################################################
nodeSet    = odb.rootAssembly.nodeSets['DOMAIN']
coordNodal = lastFrame.fieldOutputs['COORD'].getSubset(region=nodeSet, position=NODAL)
dispOutput = lastFrame.fieldOutputs['U'].getSubset(region=nodeSet)
strainElem = lastFrame.fieldOutputs['E'].getSubset(region = nodeSet, position=NODAL)

nNode     = len(coordNodal.values)
totalData = zeros((nNode, 9))

for j in range(nNode):
    totalData[j,0]    = coordNodal.values[j].nodeLabel
    totalData[j,1:3]  = coordNodal.values[j].data
    totalData[j,3:5]  = dispOutput.values[j].data
    totalData[j,5:9] = strainElem.values[j].data
    
# Data Array <-- Ordering with Node Label
#     0       //  1  //  2  //  3  //   4  //   5  //    6  //   4   //   5   //    6   //   7   //   8   //    9   //
# Node Label  //  X  //  Y  //  Z  //  Ux  //  Uy  //   Uz  //  exx  //  eyy  //   ezz  //  exy  //  exz  //   eyz  //

savetxt('nodalData_Temp.dat', totalData, fmt='%15.7e', delimiter='\t')

odb.close()