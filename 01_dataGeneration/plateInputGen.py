# -*- coding: mbcs -*-
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import random
from numpy import array
import numpy as np
from math import cos, sin, radians

def abaqusInputGen(elemType, bcFlag, nodeArray, meshSize, loadType, loadEq, angVector):
    '''
    # Geometry Generation
    
      Node 4------ Node 3
         |            |
         |            |
      Node 1------ Node 2
    
    * Default Node List
    - node_1 = [112, 0]
    - node_2 = [195, 0]
    - node_3 = [99, 200]
    - node_4 = [69, 200]
    
    Description: This code is only applicable to generate the vertical member. 
                 Additional function will be added to generate the lateral member. 
                 Developed data will be utilized multiple timest by invert results
                 for lateral and veritcal directions.
    '''
    ###############################################################################
    # INPUT PARAMETERS
    ###############################################################################
    # FLAG INFORMATION
    '''
    OPTIONS:
    Elem Type : TRI, QUAD
    BC Type   : FIXED, PINNED
    Load Type : Point, Distributed
    Load Angle: 45deg, 90deg, 
    '''
    
    # GEOMETRY 
    node_1   = nodeArray[0,:]
    node_2   = nodeArray[1,:]
    node_3   = nodeArray[2,:]
    node_4   = nodeArray[3,:]
    vertAmp   = 10
    
    ###############################################################################
    # GEOMETRY GENERATION
    ###############################################################################
    mdb.models.changeKey(fromName='Model-1', toName='PLATE')
    mdb.models['PLATE'].ConstrainedSketch(name='__profile__', sheetSize=500.0)
    mdb.models['PLATE'].sketches['__profile__'].Line(point1=(node_1[0], node_1[1]), point2=(
        node_2[0], node_2[1]))
    mdb.models['PLATE'].sketches['__profile__'].Line(point1=(node_2[0], node_2[1]), point2=(
        node_3[0], node_3[1]))
    mdb.models['PLATE'].sketches['__profile__'].Line(point1=(node_3[0], node_3[1]), point2=(
        node_4[0], node_4[1]))
    mdb.models['PLATE'].sketches['__profile__'].Line(point1=(node_4[0], node_4[1]), point2=(
        node_1[0], node_1[1]))
    mdb.models['PLATE'].Part(dimensionality=TWO_D_PLANAR, name='SHELL', type=
        DEFORMABLE_BODY)
    mdb.models['PLATE'].parts['SHELL'].BaseShell(sketch=
        mdb.models['PLATE'].sketches['__profile__'])
    del mdb.models['PLATE'].sketches['__profile__']
    ###############################################################################
    # CUTTING THE PLATE WITH RESPECT TO THE TRAINING WINDOW SIZE
    ###############################################################################
    mdb.models['PLATE'].parts['SHELL'].DatumPlaneByPrincipalPlane(offset=50.0, 
        principalPlane=XZPLANE)
    mdb.models['PLATE'].parts['SHELL'].DatumPlaneByPrincipalPlane(offset=150.0, 
        principalPlane=XZPLANE)
    mdb.models['PLATE'].parts['SHELL'].PartitionFaceByDatumPlane(datumPlane=
        mdb.models['PLATE'].parts['SHELL'].datums[3], faces=
        mdb.models['PLATE'].parts['SHELL'].faces.getSequenceFromMask(('[#1 ]', ), 
        ))
    mdb.models['PLATE'].parts['SHELL'].PartitionFaceByDatumPlane(datumPlane=
        mdb.models['PLATE'].parts['SHELL'].datums[2], faces=
        mdb.models['PLATE'].parts['SHELL'].faces.getSequenceFromMask(('[#1 ]', ), 
        ))
    ###############################################################################
    # MATERIAL PROPERTIES
    ###############################################################################
    mdb.models['PLATE'].Material(name='STEEL')
    mdb.models['PLATE'].materials['STEEL'].Elastic(table=((200000000000.0, 0.3), ))
    mdb.models['PLATE'].HomogeneousSolidSection(material='STEEL', name='STEEL', 
        thickness=1.0)
    mdb.models['PLATE'].parts['SHELL'].Set(faces=
        mdb.models['PLATE'].parts['SHELL'].faces.getSequenceFromMask(('[#7 ]', ), )
        , name='WHOLE')
    mdb.models['PLATE'].parts['SHELL'].SectionAssignment(offset=0.0, offsetField=''
        , offsetType=MIDDLE_SURFACE, region=
        mdb.models['PLATE'].parts['SHELL'].sets['WHOLE'], sectionName='STEEL', 
        thicknessAssignment=FROM_SECTION)
    mdb.models['PLATE'].rootAssembly.DatumCsysByDefault(CARTESIAN)
    mdb.models['PLATE'].rootAssembly.Instance(dependent=ON, name='SHELL-1', part=
        mdb.models['PLATE'].parts['SHELL'])
    ###############################################################################
    # STEP ASSIGN
    ###############################################################################
    mdb.models['PLATE'].StaticStep(name='STATIC', previous='Initial')
    mdb.models['PLATE'].fieldOutputRequests['F-Output-1'].setValues(variables=('S', 
        'E', 'U', 'ENER', 'EVOL', 'COORD'))
    ###############################################################################
    # BC REGION DEFINE
    ###############################################################################
    mdb.models['PLATE'].rootAssembly.Set(name='BOTTOM_NODES', vertices=
        mdb.models['PLATE'].rootAssembly.instances['SHELL-1'].vertices.getSequenceFromMask(
        ('[#c ]', ), ))
    mdb.models['PLATE'].rootAssembly.Set(edges=
        mdb.models['PLATE'].rootAssembly.instances['SHELL-1'].edges.getSequenceFromMask(
        ('[#4 ]', ), ), name='BOTTOM_EDGES')
    mdb.models['PLATE'].rootAssembly.Set(edges=
        mdb.models['PLATE'].rootAssembly.instances['SHELL-1'].edges.getSequenceFromMask(
        ('[#100 ]', ), ), name='TOP_EDGES')
    mdb.models['PLATE'].rootAssembly.Set(name='TOP_NODE', vertices=
        mdb.models['PLATE'].rootAssembly.instances['SHELL-1'].vertices.getSequenceFromMask(
        ('[#c0 ]', ), ))
    ###############################################################################
    # MESH GENERATION
    ###############################################################################
    if elemType == 'QUAD':
        mdb.models['PLATE'].parts['SHELL'].setMeshControls(elemShape=QUAD, regions=
            mdb.models['PLATE'].parts['SHELL'].faces.getSequenceFromMask(('[#7 ]', ), )
            , technique=STRUCTURED)    
    elif elemType == 'TRI':
        mdb.models['PLATE'].parts['SHELL'].setMeshControls(elemShape=TRI, 
            regions= mdb.models['PLATE'].parts['SHELL'].faces.getSequenceFromMask((
            '[#7 ]', ), ))    
    else:
        pass
    
    mdb.models['PLATE'].parts['SHELL'].setElementType(elemTypes=(ElemType(
        elemCode=CPE4R, elemLibrary=STANDARD, secondOrderAccuracy=OFF, 
        hourglassControl=DEFAULT, distortionControl=DEFAULT), ElemType(
        elemCode=CPE3, elemLibrary=STANDARD)), regions=(
        mdb.models['PLATE'].parts['SHELL'].faces.getSequenceFromMask(('[#7 ]', ), 
        ), ))
    mdb.models['PLATE'].parts['SHELL'].seedPart(deviationFactor=0.1, minSizeFactor=
        0.1, size=meshSize)
    
    mdb.models['PLATE'].parts['SHELL'].generateMesh()
    mdb.models['PLATE'].rootAssembly.regenerate()
    ###############################################################################
    # OPTIMIZATION
    ###############################################################################
    mdb.models['PLATE'].rootAssembly.Set(faces=
        mdb.models['PLATE'].rootAssembly.instances['SHELL-1'].faces.getSequenceFromMask(
        ('[#2 ]', ), ), name='DOMAIN')
    mdb.models['PLATE'].TopologyTask(materialInterpolationTechnique=SIMP, name=
        'TOP_OPT', region=mdb.models['PLATE'].rootAssembly.sets['DOMAIN'])
    mdb.models['PLATE'].optimizationTasks['TOP_OPT'].SingleTermDesignResponse(
        drivingRegion=None, identifier='STRAIN_ENERGY', name='SE', operation=SUM, 
        region=mdb.models['PLATE'].rootAssembly.sets['DOMAIN'], stepOptions=())
    mdb.models['PLATE'].optimizationTasks['TOP_OPT'].SingleTermDesignResponse(
        drivingRegion=None, identifier='VOLUME', name='VOL', operation=SUM, region=
        mdb.models['PLATE'].rootAssembly.sets['DOMAIN'], stepOptions=())
    mdb.models['PLATE'].optimizationTasks['TOP_OPT'].ObjectiveFunction(name=
        'Objective-1', objectives=((OFF, 'SE', 1.0, 0.0, ''), ))
    mdb.models['PLATE'].optimizationTasks['TOP_OPT'].OptimizationConstraint(
        designResponse='VOL', name='Opt-Constraint-1', restrictionMethod=
        RELATIVE_LESS_THAN_EQUAL, restrictionValue=0.5)
    ###############################################################################
    # BOUNDARY CONDITIONS
    ###############################################################################
    if bcFlag == 'Fixed':
        mdb.models['PLATE'].EncastreBC(createStepName='STATIC', localCsys=None
            , name='BC-1', region=
            mdb.models['PLATE'].rootAssembly.sets['BOTTOM_EDGES'])
    elif bcFlag == 'Pinned':
        mdb.models['PLATE'].PinnedBC(createStepName='STATIC', localCsys=None, name=
            'BC-1', region=mdb.models['PLATE'].rootAssembly.sets['BOTTOM_NODES'])    
    else:
        pass
    ###############################################################################
    # LOAD CONDITION
    ###############################################################################
    mdb.models['PLATE'].ExpressionField(description='', expression=loadEq,
                                        localCsys=None, name='AnalyticalField-1')
    mdb.models['PLATE'].rootAssembly.Surface(name='TOP_SURF', side1Edges=
        mdb.models['PLATE'].rootAssembly.instances['SHELL-1'].edges.getSequenceFromMask(
        ('[#100 ]', ), ))
    mdb.models['PLATE'].SurfaceTraction(createStepName='STATIC', directionVector=angVector, distributionType=UNIFORM, field='', 
        localCsys=None, magnitude=vertAmp, name='TOPLOAD', region=
        mdb.models['PLATE'].rootAssembly.surfaces['TOP_SURF'], traction=GENERAL)
    mdb.models['PLATE'].loads['TOPLOAD'].setValues(distributionType=FIELD, field=
            'AnalyticalField-1')
    ###############################################################################
    # JOB DEFINE
    ###############################################################################
    mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF, 
        explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF, 
        memory=90, memoryUnits=PERCENTAGE, model='PLATE', modelPrint=OFF, 
        multiprocessingMode=DEFAULT, name='PLATE_STATIC', nodalOutputPrecision=
        SINGLE, numCpus=1, numDomains=1, numGPUs=0, queue=None, resultsFormat=ODB, 
        scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
    mdb.OptimizationProcess(dataSaveFrequency=OPT_DATASAVE_SPECIFY_CYCLE, 
        description='', maxDesignCycle=50, model='PLATE', name='TOP_OPT', 
        odbMergeFrequency=2, prototypeJob='TOP_OPT-Job', saveInitial=False, task=
        'TOP_OPT')
    mdb.optimizationProcesses['TOP_OPT'].Job(atTime=None, getMemoryFromAnalysis=
        True, memory=90, memoryUnits=PERCENTAGE, model='PLATE', 
        multiprocessingMode=DEFAULT, name='TOP_OPT-Job', numCpus=1, numDomains=1, 
        numGPUs=0, queue=None, waitHours=0, waitMinutes=0)
    #####################################################################################
    # JOB INPUT
    #####################################################################################
    mdb.jobs['PLATE_STATIC'].writeInput()
    mdb.optimizationProcesses['TOP_OPT'].writeParAndInputFiles()    

def pickNodes(mapSize):
    
    '''
    Pick the two nodes on two edges
    Top    ----------
           |        |
    Left   |        | Right
           |        | 
    Bottom ----------   
    '''
    
    sampleList_1 = random.sample(range(0,mapSize), 2)
    sampleList_2 = random.sample(range(0,mapSize), 2)
    sampleList_1.sort()
    sampleList_2.sort()
    
    node_1       = [sampleList_1[0], 0]
    node_2       = [sampleList_1[1], 0]
    node_3       = [sampleList_1[1], mapSize]
    node_4       = [sampleList_1[0], mapSize]
    
    nodeArray    = array([node_1, node_2, node_3, node_4])
    
    return nodeArray

def loadEqGen(nodeArray, loadType):
    '''
      X1 *----------* X2
          \        /
           \      /
            *----*
    '''
    x1 = nodeArray[3,0]
    x2 = nodeArray[2,0]
    xLen = int(x2-x1)
    
    if loadType == 'Point':
        loc = random.sample(range(x1,x2),1)
        loadEq = 'floor(sin(X*pi/' + str(2*loc[0]) + ')*1.05)'
    else:
        randValueList_1 = random.sample(range(2*xLen, 4*xLen), 2)
        randValueList_2 = random.sample(range(-2*xLen, 2*xLen),2)
        
        loadEq = 'sin(X*pi/' + str(randValueList_1[0]) + '+(' + str(randValueList_2[0]) + ')' + ')' + '+' + \
                 'cos(X*pi/' + str(randValueList_1[1]) + '+(' + str(randValueList_2[1]) + ')' + ')'
    return loadEq

def angCalc(rangeLimit):
    randomAngle = radians(random.sample(range(rangeLimit[0],rangeLimit[1]),1)[0])
    x = sin(randomAngle)
    y = cos(randomAngle)
    angleVector = ((0, 0, 0), (x, y, 0))
    return angleVector

if __name__ == '__main__':
    mapSize      = 200
    meshSize     = 2.0
    
    randInt      = np.random.randint(2, size=1)
    elemTypeList = ['TRI','QUAD'] 
    elemType     = elemTypeList[randInt[0]]
    bcFlag       = 'Pinned'    # Pinned
    loadType     = 'Distributed'    # Distributed
    #loadAng      = 45
    angleLimit   = [0,360]
    angVector    = angCalc(angleLimit)
    #print(angVector)
    nodeArray    = pickNodes(mapSize)
    loadEquation = loadEqGen(nodeArray, loadType)
    abaqusInputGen(elemType, bcFlag, nodeArray, meshSize, loadType, loadEquation, angVector)