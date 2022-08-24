from odbAccess import *
from numpy import array, concatenate, zeros, genfromtxt, savetxt, sort, unique, where
import sys

odb          = openOdb(path='PLATE_STATIC.odb')
lastFrame    = odb.steps['STATIC'].frames[-1]

#############################################################################################
# GLOBAL ELEM DATA EXPORT
#############################################################################################
elemSet    = odb.rootAssembly.elementSets['DOMAIN']
coordElemAct  = lastFrame.fieldOutputs['COORD'].getSubset(region = elemSet)

nElemAct      = len(coordElemAct.values)
elemDataAct   = zeros((nElemAct, 3))

for k in range(nElemAct):
    elemDataAct[k,0]   = coordElemAct.values[k].elementLabel
    elemDataAct[k,1:3] = coordElemAct.values[k].data

# Data Array <-- Ordering with Node Label
#     0       //  1  //  2  //  3  //
# Elem Label  //  X  //  Y  //  Z  //
savetxt('elemCoord.dat', elemDataAct, fmt='%15.7e', delimiter='\t')
odb.close()