from odbAccess import *
from numpy import array, concatenate, zeros, genfromtxt, savetxt, sort, unique, where
import sys
import os
from os import listdir
from os.path import isfile, join

# Pick Actual File Name in Current Directory
currentdir   = os.getcwd()
fileName     = os.listdir(currentdir)[1]

#print(fileName)
# Main Code Set
# Open ODB file
odb          = openOdb(path=fileName)

# Create a variable that refers to the 
# last frame of the first step.

lastFrame    = odb.steps['STATIC'].frames[-1]


##################################################################################
# ELEMDATA EXPORT
##################################################################################
#elemAll      = odb.rootAssembly.elementSets['DOMAIN']

# Assign Set Region
elemSet        = odb.rootAssembly.instances['PART-1-1'].elementSets['DOMAIN']
matPropDensity = lastFrame.fieldOutputs['MAT_PROP_NORMALIZED'].getSubset(region=elemSet)

# Define outputs
nElem        = len(matPropDensity.values)
dataArray    = zeros((nElem, 2))

for i in range(nElem):
    # Data Array <-- Ordering with Node Label
    #     0       //      1     //
    # Elem Label  // MatDensity //
    dataArray[i,0] = matPropDensity.values[i].elementLabel
    dataArray[i,1] = matPropDensity.values[i].data

savetxt('matDensityData.dat', dataArray, fmt='%15.7e', delimiter='\t')

# Close ODB file
odb.close()
