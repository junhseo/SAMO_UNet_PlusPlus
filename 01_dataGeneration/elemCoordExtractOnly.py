import os
import shutil
import subprocess
import datetime
from numpy import zeros, array, linspace, genfromtxt, flip, savetxt, sort, unique, where, random, concatenate
from math import sin, cos, pi
import itertools

def coordExtract(originalDirectory, initialCaseNum, numLoadCase): 
    # Data Saving Folder Generation
    #os.mkdir('dataFolder')
    # Loop for Data Generation

    for i in range(numLoadCase):
        # Set New Folder Name
        caseNum  = int(initialCaseNum + i)
        workPath = 'Case_' + str(caseNum)
        
        # Create New Folder
        if not os.path.exists(workPath):
            os.mkdir(workPath)
        
        # Copy Files for Static Analysis
        #for filename in copyFileListStatic:
        #    shutil.copyfile(filename, os.path.join(workPath,filename))
        
        # Change Path
        os.chdir(workPath)
        
        cmd = "abaqus python elemCoordExport.py interactive"
        return_code = subprocess.call(cmd,shell=True)
        
        # Source
        currentPath   = os.getcwd()
        src_NodalFile = os.path.join(currentPath, 'elemCoord.dat')
        
        # File Copys
        shutil.copy(src_NodalFile, originalDirectory  +"/dataFolder")
        
        # Destination
        dst_NodalFile = os.path.join(originalDirectory +"/dataFolder",  'elemCoord.dat')
        
        # New File Name
        new_dst_NodalFile = os.path.join(originalDirectory +"/dataFolder", 'elemCoord_'+str(caseNum)+'.dat')
        
        # Renaming
        os.rename(dst_NodalFile, new_dst_NodalFile)
       
        ## Load Density Information after Topology Optimization
        if (i+1)%5 ==0:
            now = datetime.datetime.now()
            date_time = now.strftime("%d/%m/%Y %H:%M:%S")
            print(f'Progress {i+1}/500 // Time: {date_time}')     
            
        os.chdir(originalDirectory)


if __name__ == "__main__":
    originalDirectory = os.getcwd()
    numLoadCase       = 1000
    initialCaseNum    = 0
    coordExtract(originalDirectory, initialCaseNum, numLoadCase)
