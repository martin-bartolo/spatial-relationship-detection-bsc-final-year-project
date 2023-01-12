import numpy as np
import math
from sVOC2k_lib_util import Object
#
#
def compute_depth_average(obj, im_depth_map):
    # obj is an instance of Object (see above)
    # im_depth is a 2D depth map
    #
    obj_section = im_depth_map[obj.ymin:obj.ymax+1, obj.xmin:obj.xmax+1]
    return 1-np.mean(obj_section)
#
#
def compute_radial_weighted_average(obj, im_depth_map):
    # obj is an instance of Object (see above)
    # im_depth is a 2D depth map
    #
    obj_section = im_depth_map[obj.ymin:obj.ymax+1, obj.xmin:obj.xmax+1]
    Sy,Sx=obj_section.shape
    Ox=int(Sx/2.)
    Oy=int(Sy/2.)
    a_1=np.ones(obj_section.shape,dtype=float)
    a_x = a_1 * np.linspace(0,Sx-1,Sx).reshape([1,Sx])
    a_y = a_1*np.linspace(0,Sy-1,Sy).reshape([Sy,1])
    a_d = np.sqrt((a_x-Ox)**2.0 + (a_y-Oy)**2.0)
    normalizer =  np.mean((np.max(a_d) - a_d)*a_1)
    #
    return 1-np.mean((np.max(a_d) - a_d)*obj_section)/normalizer
#
def compute_radial_weighted_average_deprecated(obj, im_depth_map):
    # obj is an instance of Object (see above)
    # im_depth is a 2D depth map
    # arr is a numpy array
    def get_distance(_x,_y):        
        return np.sqrt((_x-Ox)**2.0+(_y-Oy)**2.0)
    #
    def compute_weighted_average(b):
        max_dist=get_distance(0,0)
        total_sum=0
        c=1  # constant
        for j in range(Sy):
            for i in range(Sx):
                total_sum += c*(max_dist-get_distance(i,j))*b[j,i] 
        return total_sum
    #
    obj_section = im_depth_map[obj.ymin:obj.ymax+1, obj.xmin:obj.xmax+1]
    Sy,Sx=obj_section.shape
    Ox=int(Sx/2.)
    Oy=int(Sy/2.)
    blank=np.ones(obj_section.shape,dtype=float)
    #
    obj_sum=compute_weighted_average(obj_section)
    norm=compute_weighted_average(blank)
    #
    #return Sx,Sy,Ox,Oy, get_distance(0,0),get_distance(Ox,Oy),get_distance(Ox+1,Oy+1),max_dist
    return  1-obj_sum/norm   
#
#
def getObjectArea(T):
    # Computes the bounding box area in terms of pixels squared
    # and returns value as type float
    # T is an instance of type Object
    objectArea=-1.0    
    objectArea = (float(T.xmax)-float(T.xmin))*(float(T.ymax)-float(T.ymin))
    return objectArea
#    
def getBoxDiagonal(sizeX,sizeY): 
    # compute diagonal of image in pixels
    # sizeX is x dimension of image box
    # sizeY is y dimension of image box
    boxDiagonal=-1.0    
    boxDiagonal = (float(sizeX)**2.0+float(sizeY)**2.0)**0.5
    return boxDiagonal
#
def getObjectDiagonal(T): 
    # compute diagonal of an object Bounding Box in pixels
    # T is an instance of type Object
    objDiagonal=-1.0    
    objDiagonal = ((float(T.xmax)-float(T.xmin))**2.0+ \
                   (float(T.ymax)-float(T.ymin))**2.0)**0.5
    return objDiagonal
#
def getUnionBB(Ta,Tb):
    # returns the limits of the bounding box that encapsulates 
    # the union of the two objects
    # referred to as "enclosing box" in Ramisa etal 
    # the limits are defined as type Object with label="Union"
    Tu = Object(label='Union')
    Tu.xmin = min(Ta.xmin,Tb.xmin)
    Tu.xmax = max(Ta.xmax,Tb.xmax)
    Tu.ymin = min(Ta.ymin,Tb.ymin)
    Tu.ymax = max(Ta.ymax,Tb.ymax)
    return Tu
#
def getCentroids(Ta,Tb):
    # Computes centroids for two objects; 
    # Ta and Tb which are of type Object
    # Dimensions are pixels
    # return values as type float
    c1=c2=-1.0
    c1 = ((float(Ta.xmax)-float(Ta.xmin))/2.0+float(Ta.xmin), 
          (float(Ta.ymax)-float(Ta.ymin))/2.0+float(Ta.ymin))
    c2 = ((float(Tb.xmax)-float(Tb.xmin))/2.0+float(Tb.xmin), 
          (float(Tb.ymax)-float(Tb.ymin))/2.0+float(Tb.ymin))
    return c1,c2
#
#            
def getDistanceBetweenCentroids(Ta,Tb):
    # Computes distance in pixels between centroids 
    # Ta and Tb are of type Object
    # Return value as type float
    # normalise by diagonal
    distance=-1.0
    c1,c2=getCentroids(Ta,Tb)
    distance = (float(c1[0]-c2[0])**2.0 + float(c1[1]-c2[1])**2.0)**0.5 
    return distance
#
def getDistanceBetweenCentroids_norm_unionBB(Ta,Tb,Tu):
    # Computes distance in pixels between centroids 
    # Ta and Tb are of type Object
    # Return value as type float
    # normalise by diagonal
    distance=-1.0
    c1,c2=getCentroids(Ta,Tb)
    distance = ((float(c1[0]-c2[0])/(Tu.xmax-Tu.xmin))**2.0 +\
                (float(c1[1]-c2[1])/(Tu.ymax-Tu.ymin))**2.0)**0.5 
    return distance
#
#
def getAreaOverlap(Tx,Ty,Ta,Tb):
    # Calculates the overlapping area in pixels squared of two bounding boxes
    # Value is returned as type float
    # Ta,Tb are the two objects of type Object
    # Tx and Ty are the width and height of image
    #
    areaOverlap=-1.0
    # P1 is a numpy array of zeros; size of image
    P1 = np.zeros((int(Tx),int(Ty)), dtype=np.int)
    # fill in P1 with ones where Ta is located
    P1[int(Ta.xmin):int(Ta.xmax), int(Ta.ymin):int(Ta.ymax)]=1
    #
    # P2 is a numpy array of zeros; size of image
    P2 = np.zeros((int(Tx),int(Ty)), dtype=np.int) 
    # fill in P2 with ones where Tb is located
    P2[int(Tb.xmin):int(Tb.xmax), int(Tb.ymin):int(Tb.ymax)]=1
    #
    areaOverlap = float(np.sum(P1*P2))
    return areaOverlap
#   
def getDistanceSizeRatio(Ta,Tb):
    # Computes ratio of distance in between centroids and 
    # approximate widths of bounding boxes
    # Ta and Tb are of type Object
    dsr=-1.0
    dbc = getDistanceBetweenCentroids(Ta,Tb)
    #Estimate Widths assuming object is square    
    a1 = getObjectArea(Ta)**0.5/2.0
    a2 = getObjectArea(Tb)**0.5/2.0
    dsr = dbc/(a1+a2)
    return dsr
#
#
def getRelativePosition(Ta,Tb):
    # Computes position of Object Ta relative to Object Tb
    # for example : Ta is north of Tb: Ta is above Tb
    # Ta and Tb are of type Object
    rp=-1
    c1,c2=getCentroids(Ta,Tb)
    Dy = -(c1[1]-c2[1])
    Dx = (c1[0]-c2[0])
    if Dx==0:
        Dx = 0.00000001
    theta = math.atan(Dy/Dx)/3.142*180.0
    if Dy>=0.0 and Dx<0.0:
        theta = theta +180
    if Dy<0.0 and Dx<0.0:
        theta = theta +180
    if Dy<0.0 and Dx>=0.0:
        theta = 360+theta        
    #
    if (theta>=315.0) or theta<45.0:
        rp=0
    if (theta>=45.0 and theta<135.0):
        rp=1
    if (theta>=135.0 and theta<225.0):
        rp=2
    if (theta>=225.0 and theta<315.0):
        rp=3
        #
    return rp
#
#
def computeInvFeature(Ta,Tb):
    # Ta and Tb are of type Object
    #feature=[0.0,0.0,0.0,0.0,0.0,0.0]
    feature = [0.0]*6
    normaliserX=(float(Ta.xmax)-float(Ta.xmin))
    normaliserY=(float(Ta.ymax)-float(Ta.ymin))
    feature[0]=(float(Tb.xmin)-float(Ta.xmin))/normaliserX
    feature[1]=(float(Tb.xmax)-float(Ta.xmin))/normaliserX
    feature[2]=(float(Tb.ymin)-float(Ta.ymin))/normaliserY
    feature[3]=(float(Tb.ymax)-float(Ta.ymin))/normaliserY
    #calculate aspect ratios = width/height
    feature[4]=(float(Ta.xmax)-float(Ta.xmin))/(float(Ta.ymax)-float(Ta.ymin))        
    feature[5]=(float(Tb.xmax)-float(Tb.xmin))/(float(Tb.ymax)-float(Tb.ymin))        
    return feature
#
def getEuclideanVector(iX, iY, Ta, Tb):
    # Compute vector (x,y) that gives the direction of the shortest
    # distance in between Ta and Tb
    # Ta and Tb are of type Object
    deltaX = -1.0
    deltaY = -1.0
    if getAreaOverlap(iX, iY, Ta, Tb)>0:
        deltaX=0.0
        deltaY=0.0
    else:
        if (Tb.xmin-Ta.xmax)>=0: #East
            deltaX = float(Tb.xmin)-float(Ta.xmax)
            if (Ta.ymin-Tb.ymax)>=0: # SE
                deltaY = float(Ta.ymin)-float(Tb.ymax)
                #distance=1.3
            elif (Tb.ymin-Ta.ymax)>=0: # NE
                deltaY = float(Tb.ymin)-float(Ta.ymax)
                #distance=1.1
            else:
                deltaY=0.0
                #distance=1.2
        #
        elif (Ta.xmin-Tb.xmax)>=0: #West
            deltaX=float(Ta.xmin)-float(Tb.xmax)
            if (Ta.ymin-Tb.ymax)>=0: # SW
                deltaY = float(Ta.ymin)-float(Tb.ymax)
                #distance=2.3
            elif (Tb.ymin-Ta.ymax)>=0: # NE
                deltaY = float(Tb.ymin)-float(Ta.ymax)
                #distance=2.1
            else:
                deltaY=0.0
                #distance=2.2
        #                
        elif (Tb.ymin-Ta.ymax)>=0:
            deltaX = 0.0
            deltaY = float(Tb.ymin) - float(Ta.ymax)
            #distance=3.0
        #
        elif (Ta.ymin-Tb.ymax)>=0:
            deltaX = 0
            deltaY = float(Ta.ymin) - float(Tb.ymax)
            #distance=4.0
    #
    return deltaX, deltaY
#
def getUnitEuclideanVector(iX, iY, Ta, Tb):
    deltaX,deltaY=getEuclideanVector(iX,iY,Ta,Tb)
    mag = math.sqrt(deltaX**2.0 + deltaY**2.0)
    if mag==0:
        mag=1
    return (deltaX/mag, deltaY/mag)
    
def getEuclideanDistancePixels(iX,iY,Ta,Tb):
    deltaX=-1.0
    deltaY=-1.0
    if getAreaOverlap(iX, iY, Ta, Tb)>0:
        deltaX=0.0
        deltaY=0.0
    else:
        deltaX,deltaY=getEuclideanVector(iX,iY,Ta,Tb)
    #
    return (deltaX**2.0 + deltaY**2.0)**0.5
 #   
def getEuclideanDistanceUnion(iX,iY,Ta,Tb):
    deltaX=-1.0
    deltaY=-1.0
    if getAreaOverlap(iX, iY, Ta, Tb)>0:
        deltaX=0.0
        deltaY=0.0
    else:
        deltaX,deltaY = getEuclideanVector(iX,iY,Ta,Tb)
    #
    Tc = getUnionBB(Ta,Tb)
    deltaX = deltaX/float(Tc.xmax-Tc.xmin)
    deltaY = deltaY/float(Tc.ymax-Tc.ymin)
    return (deltaX**2.0 + deltaY**2.0)**0.5
#
def getEuclideanDistanceImage(iX,iY,Ta,Tb):
    deltaX=-1.0
    deltaY=-1.0
    if getAreaOverlap(iX, iY, Ta, Tb)>0:
        deltaX=0.0
        deltaY=0.0
    else:
        deltaX,deltaY=getEuclideanVector(iX,iY,Ta,Tb)
    #
    deltaX = deltaX/float(iX)
    deltaY = deltaY/float(iY)
    return (deltaX**2.0 + deltaY**2.0)**0.5
#    
#
def getVecTrajLandNormUnionBB(Ta,Tb):
    Tu = getUnionBB(Ta,Tb)
    dX = Tu.xmax - Tu.xmin
    dY = Tu.ymax - Tu.ymin
    k0,k1=getCentroids(Ta,Tb)
    vecX = float(k1[0]-k0[0])/float(dX)
    vecY = -float(k1[1]-k0[1])/float(dY)
    return (vecX,vecY)
#
#
def getUnitVecTrajLandNormUnionBB(Ta,Tb):
    vecX,vecY = getVecTrajLandNormUnionBB(Ta,Tb)
    mag = np.sqrt(vecX**2.0 + vecY**2.0)
    if mag==0:
        mag=1
    return (vecX/mag,vecY/mag)
#
#
def getUnitVecTrajLand(Ta,Tb):
    k0,k1=getCentroids(Ta,Tb)
    vecX = float(k1[0]-k0[0])
    vecY = -float(k1[1]-k0[1])
    mag = math.sqrt(vecX**2.0 + vecY**2.0)
    if mag==0:
        mag=1
    return (vecX/mag,vecY/mag)
#
#
    
def compute_geometrical_features(imSizeX, imSizeY, obj_1, obj_2):
    #
     
    #    # STORE features in dictionary
    features = {}
    extraFeat = {}
    #
    objUnion=getUnionBB(obj_1,obj_2)
    #
    # COMPUTE AREAS
    areaWholeImage = float(imSizeX)*float(imSizeY)
    areaObj_1 = getObjectArea(obj_1)      
    areaObj_2 = getObjectArea(obj_2)
    areaUnion = getObjectArea(objUnion)
    extraFeat['areaWholeImage']=areaWholeImage
    extraFeat['areaObj_1'] = areaObj_1
    extraFeat['areaObj_2'] = areaObj_2
    extraFeat['areaUnion'] = areaUnion    
    #
    # NORMALISE AREAS
    normImageObj_1 = areaObj_1/areaWholeImage     
    normImageObj_2 = areaObj_2/areaWholeImage
    normUnionObj_1 = areaObj_1/areaUnion     
    normUnionObj_2 = areaObj_2/areaUnion
    features['AreaObj1_Norm_wt_Image']=normImageObj_1
    features['AreaObj2_Norm_wt_Image']=normImageObj_2
    features['AreaObj1_Norm_wt_Union']=normUnionObj_1
    features['AreaObj2_Norm_wt_Union']=normUnionObj_2
    #
    #AREA RATIOS     
    objAreaRatioTL = areaObj_1/areaObj_2
    objAreaRatioMM = max(areaObj_1,areaObj_2)/min(areaObj_1,areaObj_2)
    features['objAreaRatioTrajLand'] = objAreaRatioTL
    features['objAreaRatioMaxMin'] = objAreaRatioMM 
    #
    #COMPUTE DIAGONALS
    diagImage = getBoxDiagonal(imSizeX,imSizeY)
    diagUnion = getObjectDiagonal(objUnion)
    diagObj_1 = getObjectDiagonal(obj_1)
    diagObj_2 = getObjectDiagonal(obj_2)
    extraFeat['diagonalImage'] = diagImage
    extraFeat['diagonalUnion'] = diagUnion 
    extraFeat['diagonalObj_1'] = diagObj_1 
    extraFeat['diagonalObj_2'] = diagObj_2 
    #
    #NORMALISE DIAGONALS BY IMAGE 
    normImageDiagImage = diagImage/diagImage
    normImageDiagUnion = diagUnion/diagImage
    normImageDiagObj_1 = diagObj_1/diagImage
    normImageDiagObj_2 = diagObj_2/diagImage
    extraFeat['ImageDiag_Norm_wt_ImageDiag'] = normImageDiagImage 
    features['UnionDiag_Norm_wt_ImageDiag'] = normImageDiagUnion 
    features['Obj_1Diag_Norm_wt_ImageDiag'] = normImageDiagObj_1 
    features['Obj_2Diag_Norm_wt_ImageDiag'] = normImageDiagObj_2 
    #
    #NORMALISE DIAGONALS BY UNION 
    normUnionDiagImage = diagImage/diagUnion
    normUnionDiagUnion = diagUnion/diagUnion
    normUnionDiagObj_1 = diagObj_1/diagUnion
    normUnionDiagObj_2 = diagObj_2/diagUnion
    features['ImageDiag_Norm_wt_UnionDiag'] = normUnionDiagImage 
    extraFeat['UnionDiag_Norm_Wt_UnionDiag'] = normUnionDiagUnion 
    features['Obj_1Diag_Norm_wt_UnionDiag'] = normUnionDiagObj_1 
    features['Obj_2Diag_Norm_wt_UnionDiag'] = normUnionDiagObj_2 
    #
    #COMPUTE RATIO OF DIAGONALS
    objDiagRatioTL = diagObj_1/diagObj_2
    objDiagRatioMM = max(diagObj_1,diagObj_2)/min(diagObj_1,diagObj_2)
    features['objDiagonalRatioTL'] = objDiagRatioTL
    features['objDiagonalRatioMaxMin'] = objDiagRatioMM
    #
    #COMPUTE DISTANCE BETWEEN CENTROIDS and NORMALISE
    distBetweenCentroids=getDistanceBetweenCentroids(obj_1,obj_2)
    distBtCentr_norm_wt_image_diag=distBetweenCentroids/diagImage
    distBtCentr_norm_wt_union_diag=distBetweenCentroids/diagUnion
    extraFeat['DistBetweenCentroids'] = distBetweenCentroids
    features['DistBtCentr_Norm_wt_ImageDiag'] = distBtCentr_norm_wt_image_diag
    features['DistBtCentr_Norm_wt_UnionDiag'] = distBtCentr_norm_wt_union_diag
    features['DistBtCentr_Norm_wt_UnionBB'] =\
            getDistanceBetweenCentroids_norm_unionBB(obj_1, obj_2, objUnion)
    #
    #COMPUTE OBJECT AREA OVERLAP and NORMALISE
    objAreaOverlap = getAreaOverlap(imSizeX,imSizeY, obj_1,obj_2)
    normImageAreaOverlap = objAreaOverlap/areaWholeImage
    #normUnionAreaOverlap = objAreaOverlap/areaUnion
    normUnionAreaOverlap = objAreaOverlap/(areaObj_1+areaObj_2-objAreaOverlap)
    normTotalAreaOverlap = objAreaOverlap/(areaObj_1+areaObj_2)
    normMinAreaOverlap = objAreaOverlap/min(areaObj_1,areaObj_2)
    extraFeat['objAreaOverlap']=objAreaOverlap
    features['AreaOverlap_Norm_wt_Image'] = normImageAreaOverlap
    features['AreaOverlap_Norm_wt_Union'] =  normUnionAreaOverlap
    features['AreaOverlap_Norm_wt_Total'] = normTotalAreaOverlap
    features['AreaOverlap_Norm_wt_Min'] = normMinAreaOverlap    
    #
    #COMPUTE DISTANCE-SIZE RATIO
    distanceSizeRatio = getDistanceSizeRatio(obj_1,obj_2)
    features['DistanceSizeRatio'] = distanceSizeRatio
    #
    #COMPUTE RELATIVE POSITION
    relativePosition = getRelativePosition(obj_1,obj_2)
    features['relativePosition'] = relativePosition
    #
    #COMPUTE INVARIANT FEATURES
    invFeat = computeInvFeature(obj_1,obj_2)
    features['InvFeatXminXmin'] = invFeat[0]
    features['InvFeatXmaxXmin'] = invFeat[1]
    features['InvFeatYminYmin'] = invFeat[2]
    features['InvFeatYmaxYmin'] = invFeat[3]
    features['AspectRatioObj_1'] = invFeat[4]
    features['AspectRatioObj_2'] = invFeat[5]    
    #
    #COMPUTE EUCLIDEAN DISTANCE
    EdistancePixels = getEuclideanDistancePixels(imSizeX,imSizeY,obj_1,obj_2)
    EdistanceImage = getEuclideanDistanceImage(imSizeX,imSizeY,obj_1,obj_2)
    EdistanceUnion = getEuclideanDistanceUnion(imSizeX,imSizeY,obj_1,obj_2)
    extraFeat['euclideanDistancePixels'] = EdistancePixels
    extraFeat['euclideanDistanceVectorPixels'] = getEuclideanVector(imSizeX,imSizeY,obj_1,obj_2)
    #
    unitEuclideanVector = getUnitEuclideanVector(imSizeX,imSizeY,obj_1,obj_2)
    features['unitEuclideanVector_x'] = unitEuclideanVector[0]
    features['unitEuclideanVector_y'] = unitEuclideanVector[1]    
    #
    features['EuclDist_norm_wt_ImageBB'] = EdistanceImage
    features['EuclDist_Norm_wt_UnionBB'] = EdistanceUnion
    #
    #COMPUTE vector from trajector to landmark
    VecTrajLandNormed = getVecTrajLandNormUnionBB(obj_1,obj_2)
    features['vecTrajLand_Norm_wt_UnionBB_x'] = VecTrajLandNormed[0]
    features['vecTrajLand_Norm_wt_UnionBB_y'] = VecTrajLandNormed[1]
    #
    unitVecTrajLandNormed = getUnitVecTrajLandNormUnionBB(obj_1, obj_2)
    features['unitVecTrajLand_Norm_wt_UnionBB_x'] = unitVecTrajLandNormed[0]
    features['unitVecTrajLand_Norm_wt_UnionBB_y'] = unitVecTrajLandNormed[1]
    #
    unitVecTrajLand = getUnitVecTrajLand(obj_1,obj_2)
    features['unitVecTrajLand_x'] = unitVecTrajLand[0]
    features['unitVecTrajLand_y'] = unitVecTrajLand[1]    
    #
    return features, extraFeat


## Example of an image with two objects
#imSizeX, imSizeY = [600,400]  # size of image (X,Y)
#obj_1 = Object('person',205,120,280,220,[])  
#obj_2 = Object('dog',380,95,440,150,[])   # East
#obj_2 = Object('dog',380,20,440,90,[])   # SE
#obj_2 = Object('dog',380,240,480,350,[])   # NE
##obj_2 = Object('dog', 80,95,140,150,[])   # West
##obj_2 = Object('dog', 80,20,140,90,[])   # SW
##obj_2 = Object('dog', 80,240,140,350,[])   # NW
##obj_2 = Object('dog', 180,240,440,350,[])   # N
##obj_2 = Object('dog', 180,20,440,90,[])   # S
#obj_2 = Object('dog', 215,95,440,150,[])   # Overlap
#
#feat, extras = compute_geometrical_features(imSizeX, imSizeY, obj_1, obj_2)


#printObjectAttr(obj_1,obj_1.label)
#printObjectAttr(obj_2,obj_2.label)
#
##printObjectAttr(objUnion,'Union')
#
#print "Area of whole image (pixels) = %d"%(areaWholeImage)
#print "Area of union (%s and %s) (pixels) = %d"%(obj_1.label,obj_2.label,areaUnion)
#print "Area of %s (pixels),(image), (union) = %d, %5.4f, %5.4f"% \
#        (obj_1.label, areaObj_1, normImageObj_1, normUnionObj_1)
#print "Area of %s (pixels),(image), (union) = %d, %5.4f, %5.4f"% \
#        (obj_2.label, areaObj_2, normImageObj_2, normUnionObj_2)
#print "Ratio of object areas: trajector/landmark  =", objAreaRatioTL 
#print "Ratio of object areas: max/min  =", objAreaRatioMM 
#print "Distance in between centroids (pixels),(normPerImDiag)= %d, %5.4f" % \
#        (distBetweenCentroids, normDistBetweenCentroids)
#print "Image Diagonal (pixels), (Image), (Union) = %d, %5.4f, %5.4f"% \
#        (diagImage,normImageDiagImage,normUnionDiagImage)  
#print "Union Diagonal (pixels), (Image), (Union) = %d, %5.4f, %5.4f"% \
#        (diagUnion,normImageDiagUnion,normUnionDiagUnion)  
#print "Trajector Diagonal (pixels), (Image), (Union) = %d, %5.4f, %5.4f"% \
#        (diagObj_1, normImageDiagObj_1, normUnionDiagObj_1)  
#print "Landmark Diagonal (pixels), (Image), (Union) = %d, %5.4f, %5.4f"% \
#        (diagObj_2, normImageDiagObj_2, normUnionDiagObj_2)  
#print "Ratio of object diagonals: trajector/landmark  =", objDiagRatioTL
#print "Ratio of object diagonals: max/min  =", objDiagRatioMM
#
#print "Area overlap (pixels), (Image), (Union)= %d, %5.4f, %5.4f" % \
#        (objAreaOverlap, normImageAreaOverlap, normUnionAreaOverlap) 
#print "Area overlap (pixels), (total), (minimum) = %d, %5.4f, %5.4f" % \
#        (objAreaOverlap, normTotalAreaOverlap, normMinAreaOverlap)
#print "Distance - Size Ratio = %8.4f" % distanceSizeRatio
#print "Relative position of Trajector from Landmark = ",relativePosition
#print "Invariant Features = %6.3f, %6.3f, %6.3f, %6.3f" % \
#        (invFeat[0], invFeat[1], invFeat[2], invFeat[3]) 
#print "Aspect ratio for %s : %6.3f" % (obj_1.label, invFeat[4])
#print "Aspect ratio for %s : %6.3f" % (obj_2.label, invFeat[5])
#print "Euclidean Distance (pixel), (Image), (Union) = %7.3f, %7.3f, %7.3f" % \
#        (EdistancePixels,EdistanceImage, EdistanceUnion) 
#print "Vector from Trajector to Landmark = (%7.3f, %7.3f)"% \
#        (vectorTL[0],vectorTL[1])
#
##print "Distance-Size Ratio = ", distanceSizeRatio
##print "Position relative to first object = ", relativePosition
###word.append(xx)






















