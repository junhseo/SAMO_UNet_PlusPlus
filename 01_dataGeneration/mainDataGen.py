import os
import shutil
import subprocess
import datetime
from numpy import zeros, array, linspace, genfromtxt, flip, savetxt, sort, unique, where, random, concatenate
from math import sin, cos, pi
import itertools

#def loadCaseGen(nodeNum, angNum, volNum):
    #firstNodeList = [i+1 for i in range(nodeNum)]
    #angList       = [-pi/2 + pi/8 * j for j in range(angNum)]
    #volList       = [0.3 + 0.05*k for k in range(volNum)]
    #combList      = list(itertools.product(firstNodeList, angList, volList))
    #combArray     = array(combList)
    #return combArray

def topologyOptimization(originalDirectory, copyFileListStatic, copyFileListTopOpt, numLoadCase):
    def inputStaticUpdate(fileName):
        a_file = open(fileName, "r")
        b_file = open(fileName, "r")
        
        text   = '*Output, field'
        for position, line in enumerate(b_file):
            if text in line:
                loc = position
            else:
                pass
        list_of_lines = a_file.readlines()
        list_of_lines[loc+3] = "*Element Output, POSITION=AVERAGED AT NODES \n"
        list_of_lines[loc+4] = " E, S \n"
        list_of_lines[loc+5] = "*Element Output, directions=YES \n"
        list_of_lines[loc+6] = " COORD, E, S, EVOL \n"
        list_of_lines[loc+7] = "** \n"
        list_of_lines[loc+8] = "*Output, history, variable=PRESELECT \n"
        list_of_lines[loc+9] = "*End Step "        
        a_file = open(fileName, "w")
        a_file.writelines(list_of_lines)
        a_file.close()
    
    def inputTopOptUpdate(fileName):
        a_file = open(fileName, "r")
        b_file = open(fileName, "r")
        text   = '*Output, field'
        for position, line in enumerate(b_file):
            if text in line:
                loc = position
            else:
                pass
        list_of_lines = a_file.readlines()
        print(loc)
        list_of_lines[int(loc+3)] = "*Element Output, POSITION=AVERAGED AT NODES \n"
        list_of_lines[loc+4] = " E, S \n"
        list_of_lines[loc+5] = "*Element Output, directions=YES \n"
        list_of_lines[loc+6] = " COORD, E, S, EVOL \n"
        list_of_lines[loc+7] = "** \n"
        list_of_lines[loc+8] = "*Output, history, variable=PRESELECT \n"
        list_of_lines[loc+9] = "*End Step "        
        a_file = open(fileName, "w")
        a_file.writelines(list_of_lines)
        a_file.close()    
    
    def paramFileUpdate(fileName, volConst):
        a_file = open(fileName, "r")
        list_of_lines = a_file.readlines()
        list_of_lines[58] = "  LE_VALUE = " + str(volConst) +"\n"
        a_file = open(fileName, "w")
        a_file.writelines(list_of_lines)
        a_file.close()    
    
    # Data Saving Folder Generation
    os.mkdir('dataFolder')

    # Loop for Data Generation
    for i in range(numLoadCase):
        # Set New Folder Name
        workPath = 'Case_' + str(i+startNum)
        
        # Create New Folder
        if not os.path.exists(workPath):
            os.mkdir(workPath)
        
        # Copy Files for Static Analysis
        for filename in copyFileListStatic:
            shutil.copyfile(filename, os.path.join(workPath,filename))
        
        # Change Path
        os.chdir(workPath)
        
        # Assgin Load Condition
        #var1 = loadCase[i,0]
        #var2 = loadCase[i,1]
        #var3 = loadCase[i,2]
        #var4 = loadCase[i,3]
        
        # Update Python Script for Analysis
        #updateLoadCase('CANTILEVER_VER00.py', var1, var2, var3, var4)
        
        # Import Case Parameters
        # firstNode = loadCase[i,0]
        # angle     = loadCase[i,1]
        # volConst  = loadCase[i,2]
        
        # Develop Input Decks and Run Topology Optimization
        cmd = "abaqus cae noGui=plateInputGen.py"
        return_code = subprocess.call(cmd,shell=True)
        
        # Input File Updates
        inputTopOptUpdate('TOP_OPT-Job.inp')
        inputStaticUpdate('PLATE_STATIC.inp')
        
        # Param File Updates
        #paramFileUpdate('TOP_OPT.par', volConst)
        
        # Run Topology Optimization Process
        cmd = 'abaqus optimization task=TOP_OPT job=TOP_OPT cpus=1 interactive'
        return_code = subprocess.call(cmd,shell=True)
        
        # Run Static Analysis
        cmd = "abaqus j=PLATE_STATIC cpus=1 interactive"
        return_code = subprocess.call(cmd,shell=True)        
        
        # Extract the Static Analysis Result
        cmd = "abaqus python staticExport.py interactive"
        return_code = subprocess.call(cmd,shell=True)
        
        if i==0:
            cmd = "abaqus python elemCoordExport.py interactive"
            return_code = subprocess.call(cmd,shell=True)
        else:
            pass

        # Source
        currentPath   = os.getcwd()
        #src_ElemFile  = os.path.join(currentPath, 'elemData_Temp.dat')
        src_NodalFile = os.path.join(currentPath, 'nodalData_Temp.dat')
        
        # File Copys
        #shutil.copy(src_ElemFile, originalDirectory  +"/dataFolder")
        shutil.copy(src_NodalFile, originalDirectory  +"/dataFolder")
        
        # Destination
        #dst_ElemFile  = os.path.join(originalDirectory +"/dataFolder",  'elemData_Temp.dat')
        dst_NodalFile = os.path.join(originalDirectory +"/dataFolder",  'nodalData_Temp.dat')
        
        # New File Name
        #new_dst_ElemFile  = os.path.join(originalDirectory +"/dataFolder",  'elemData_'+str(i+startNum)+'.dat')
        new_dst_NodalFile = os.path.join(originalDirectory +"/dataFolder", 'nodalData_'+str(i+startNum)+'.dat')
        
        # Renaming
        #os.rename(dst_ElemFile, new_dst_ElemFile)
        os.rename(dst_NodalFile, new_dst_NodalFile)
        
        ###################################################################################
        ###########################   Topology Opt Postprocess ############################
        ###################################################################################
        
        # Go to Topology Optimization Result
        topOptPath = 'TOP_OPT/SAVE.ODB'
        
        # Copy Files for Topology Optimization Postprocess
        shutil.copyfile('matDensityExport.py', os.path.join(topOptPath, 'matDensityExport_ver00.py'))        
        # Extract the Topology Optimization Result
        os.chdir(topOptPath)
        cmd = "abaqus python matDensityExport.py interactive"
        return_code = subprocess.call(cmd,shell=True)
        
        # SOURCE 
        currentPathMat = os.getcwd()
        src_matDenFile = os.path.join(currentPathMat, 'matDensityData.dat')
        # FILE COPYS
        shutil.copy(src_matDenFile, originalDirectory + '/dataFolder')
        # DESTINATION
        dst_matDenFile = os.path.join(originalDirectory + '/dataFolder', 'matDensityData.dat')
        # NEW NAME
        new_dst_matDenFileName = os.path.join(originalDirectory +'/dataFolder',  'matDen_'+str(i+startNum)+'.dat')
        # Renaming
        os.rename(dst_matDenFile,  new_dst_matDenFileName)     
        ## Load Density Information after Topology Optimization
        if (i+1)%5 ==0:
            now = datetime.datetime.now()
            date_time = now.strftime("%d/%m/%Y %H:%M:%S")
            print(f'Progress {i+1}/50 // Time: {date_time}')     
            
        os.chdir(originalDirectory)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
if __name__ == "__main__":
    originalDirectory  = os.getcwd()
    copyFileListStatic = list(genfromtxt('copyFileListStatic.dat',dtype=str))    
    copyFileListTopOpt = genfromtxt('copyFileListTopOpt.dat',dtype=str)
    numLoadCase        = 1000
    startNum           = 0
    topologyOptimization(originalDirectory, copyFileListStatic,copyFileListTopOpt, numLoadCase)