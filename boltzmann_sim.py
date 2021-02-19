"""
    A simple Lattive Boltzmann fluid simulator
    Copyright (C) 2021 Amélia O. F. da S.
    <a.mellifluous.one@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
from scipy import signal
import cv2

size = 200
lattices = np.zeros((9,size,size))

#lattices[5][4,4]=1
#lattices[7,int(size/2)-100:int(size/2)+50,int(size/2)-50:int(size/2)-49]=10
#lattices[1,int(size/2)-50:int(size/2)+100,int(size/2)+50:int(size/2)+51]=10

walls = np.zeros((size,size))

#Square Wall
walls[int(size/2)-70:int(size/2)+70,int(size/2)-70:int(size/2)+70]=1

sources = np.zeros((9,size,size))
#sources[5][0,:]=3
sources[7][:,0]=3
#sources[5][-1,:]=3
#sources[1][:,-1]=3
sources[1,int(size/2):int(size/2)+50,int(size/2):int(size/2)+1]=3
#sources[7,int(size/2)-50:int(size/2),int(size/2):int(size/2)+1]=3
#sources[[0,1,2,3,5,6,7],int(size/2),int(size/2)]=10
#x,y = np.meshgrid(np.arange(size),np.arange(size))
#x=x-int(size/2)
#y=y-int(size/2)
#circle = np.sqrt(x**2+y**2)
#circle = circle<30
#sources[:,circle]=10
#sources[0]=0

directions = np.array([
     [[[1,size],[1,size],[0,size-1],[0,size-1]],[[0,size],[1,size],[0,size],[0,size-1]],[[0,size-1],[1,size],[1,size],[0,size-1]]],
     [[[1,size],[0,size],[0,size-1],[0,size]]  ,[       0,       0,       0,         0],[[0,size-1],[0,size],[1,size],[0,size]]],
     [[[1,size],[0,size-1],[0,size-1],[1,size]],[[0,size],[0,size-1],[0,size],[1,size]],[[0,size-1],[0,size-1],[1,size],[1,size]]]
])
pdirs = np.array([[-1, 1],[ 0, 1],[ 1, 1],
                  [-1, 0],[ 0, 0],[ 1, 0],
                  [-1,-1],[ 0,-1],[ 1,-1]])
dirmoduli = np.sqrt((pdirs**2).sum(axis=1))

lv = np.zeros((size,size,2))
def step(lattices,timescale,cs=300):
    #Equilibrium calculation
    localdensity = lattices.sum(axis=0)
    dirs = np.repeat(pdirs[np.newaxis,:,:],size,axis=0)
    dirs = np.moveaxis(np.repeat(dirs[np.newaxis,:,:,:],size,axis=0),2,0)
    localvelocity = dirs * np.repeat(lattices[:,:,:,np.newaxis],2,axis=3)
    #Tensor of point velocities for each lattice point
    localvelocity = np.nan_to_num(localvelocity.sum(axis=0)/np.repeat(localdensity[:,:,np.newaxis],2,axis=2))
    lv=np.copy(localvelocity)
    localvsquared = (localvelocity * localvelocity).sum(axis=2)

    equilibrium = np.zeros(lattices.shape)
    for i in range(9):
        dirlv = (dirs[i]*localvelocity).sum()
        equilibrium[i] = (1/32)*localdensity*(1+(dirlv/cs**2)+(dirlv**2/(cs**4*2)-localvsquared/(cs**2*2)))

    for i in range(9):
        #Relaxation
        lattices[i] = lattices[i] + (equilibrium[i]-lattices[i])/timescale
    directionweight = np.nan_to_num(np.copy(lattices)/lattices.sum(axis=0))
    for i in range(9):
        if i!=4:
            unraveled=np.unravel_index(i,(3,3))
            #Streaming
            d=directions[unraveled]
            #We calculate 
            lattices[i][d[2][0]:d[2][1],d[3][0]:d[3][1]] = lattices[i][d[2][0]:d[2][1],d[3][0]:d[3][1]]*(1-directionweight[i][d[0][0]:d[0][1],d[1][0]:d[1][1]])+lattices[i][d[0][0]:d[0][1],d[1][0]:d[1][1]]*(directionweight[i][d[0][0]:d[0][1],d[1][0]:d[1][1]])
            #Wall collision
    lattices+=sources
        
#Particles
def particlestep(particles):
    ret = []
    for p in particles:
        v = np.array([0,0])
        for i in range(9):
            v = (v - (pdirs[i]*lattices[i,int(p[0]),int(p[1])]*.5))
        newpos = (p[0]+v[0],p[1]+v[1])
        if not (newpos[0]<0 or newpos[0]>=size or newpos[1]<0 or newpos[1]>=size):
            ret.append(newpos)
    return ret


def gengrid(n=12*12,linesize=12):
    s = size/(linesize+1)
    g = []
    for i in range(n):
        coords = np.unravel_index(i,(linesize,linesize))
        g.append((int((coords[0]+1)*s),int((coords[1]+1)*s)))
    return g

def vellines(img,n=20**2,linesize=20,ms=10):
    s = size/(linesize+1)
    for i in range(n):
        coords = np.unravel_index(i,(linesize,linesize))
        p=(int((coords[0]+1)*s),int((coords[1]+1)*s))
        v = np.array([0,0])
        for i in range(9):
            v = (v - (pdirs[i]*lattices[i,int(p[0]),int(p[1])]*.5))
        v = v * 5
        if(np.sqrt(np.sum(v*v))>ms):
            v/=np.sqrt(np.sum(v*v))
            v*=ms
        v = (int(p[0]+v[0]),int(p[1]+v[1]))
        cv2.line(img,(p[0],size-p[1]),(v[0],size-v[1]),(0,0,0),1)

#Visualisation
name="Lattice Boltzmann Sim"
cv2.namedWindow(name,cv2.WINDOW_NORMAL)
cv2.namedWindow("Dirs",cv2.WINDOW_NORMAL)
cv2.namedWindow("FDirs",cv2.WINDOW_NORMAL)
#cv2.namedWindow("Settings")

def nothing(a):
    pass

#cv2.createTrackbar("Scale","Settings",0,255,nothing)
#cv2.setTrackbarPos("Scale","Settings",128)
#scalemulti=2
i = 0
particles = gengrid()
s=True
r=False
while 1:
    cv2.resizeWindow(name,500,500)
    while s:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        if key == ord('c'):
            s = False
    if r:
        particles = gengrid()
        r=False
    #scalemulti = 4+((cv2.getTrackbarPos("Scale","Settings")/255)-0.5)*2
    #result = (np.abs(lattices-(1/9))).sum(axis=0)
    result = lattices.sum(axis=0)
    result = result/result.max()
    result = result*255
    result = cv2.applyColorMap(result.astype(np.uint8),cv2.COLORMAP_JET)
    result = cv2.rotate(cv2.flip(result,0),cv2.ROTATE_90_CLOCKWISE)
    result = cv2.flip(result,0)
    #nshape=(int(result.shape[0]*scalemulti),int(result.shape[1]*scalemulti))
    #result = cv2.resize(result,nshape)
    #cv2.resizeWindow(name,*nshape)

    for p in particles:
        cv2.circle(result,(int(p[0]),size-int(p[1])),1,(0,0,0),1)
    vellines(result)

    cv2.imshow(name,result)
    out = np.zeros(((3*size),(3*size),3),dtype=np.uint8)
    for i in range(9):
        result = np.copy(lattices[i])
        result = result/lattices[i].max()
        result = result*255
        result = cv2.applyColorMap(result.astype(np.uint8),cv2.COLORMAP_JET)
        result = cv2.rotate(cv2.flip(result,0),cv2.ROTATE_90_CLOCKWISE)
        result = cv2.flip(result,0)
        coords = np.unravel_index(i,(3,3))
        out[size*coords[0]:size*(coords[0]+1),size*coords[1]:size*(coords[1]+1)]=result
    cv2.imshow("Dirs",out)
    out = np.zeros(((3*size),(3*size),3),dtype=np.uint8)
    for i in range(9):
        result = np.copy(np.flip(lattices,0)[i])
        result = result/lattices[i].max()
        result = result*255
        result = cv2.applyColorMap(result.astype(np.uint8),cv2.COLORMAP_JET)
        result = cv2.rotate(cv2.flip(result,0),cv2.ROTATE_90_CLOCKWISE)
        result = cv2.flip(result,0)
        coords = np.unravel_index(i,(3,3))
        out[size*coords[0]:size*(coords[0]+1),size*coords[1]:size*(coords[1]+1)]=result
    cv2.imshow("FDirs",out)

    step(lattices,25)
    particles = particlestep(particles)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        s=True
    if key == ord('r'):
        r=True