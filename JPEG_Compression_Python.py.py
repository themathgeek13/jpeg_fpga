import cv2
from pylab import *
import time

t1=time.time()

#Helper function required to display the image at regular modification intervals
def show(im):
    cv2.imshow("Image",im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Function required to pad the image prior to performing the DCT operation
# ----->>> Open to more optimization of this function
def imgPad(im):
    rows,cols,ch=im.shape
    listim=im.tolist()
    lengthPad = 16-rows%16
    widthPad = 16-cols%16

    if(lengthPad<16):
        for i in range(lengthPad):
            listim.append(listim[-1])
    larr=np.array(listim,dtype='int16')
    listim=np.transpose(larr,(1,0,2)).tolist()
    if(widthPad<16):
        for i in range(widthPad):
            listim.append(listim[-1])
    larr=np.array(listim,dtype='uint8')
    return np.transpose(larr,(1,0,2))

#Function to downsample the chrominance channel
def downsample(C):
    C=C.astype('uint16')
    Cout=np.zeros((8,8),dtype='int16')
    for i in range(8):
        for j in range(8):
            Cout[i][j]=(C[2*i][2*j]+C[2*i+1][2*j]+C[2*i,2*j+1]+C[2*i+1][2*j+1])/4
    return Cout

def al(x):
    if(x==0):
        return 1.0/sqrt(2.0)
    else:
        return 1.0

cosMat=np.zeros((8,8));
def evalcos():
    for i in range(8):
        for j in range(8):
            cosMat[i][j]=cos(i*pi*(2*j+1)/16.0)

def singleGUV(g,u,v,inv=0):
    G=0
    for x in range(8):
        for y in range(8):
            if inv==0:
                G+=0.25*al(u)*al(v)*g[x][y]*cosMat[u][x]*cosMat[v][y]
            else:
                G+=0.25*al(x)*al(y)*g[x][y]*cosMat[x][u]*cosMat[y][v]
    return G

def shift_128(subimg):
     #first of all, the input array is a numpy array of type "uint8"
    #this needs to be converted to int16 or else the shifting to center around 0 will fail
    subimg = subimg.astype('int16')
    subimg=subimg-128
    return subimg

#Function to perform a 2D DCT on the 8x8 images
def DCT_8x8_2D(subimg,shift=1, inv=0):
    if shift==1:
        subimg=shift_128(subimg)
    #now the 2D DCT can be obtained using the formula from en.wikipedia.org/wiki/JPEG
    G = np.zeros((8,8),dtype='float')
    for u in range(8):
        for v in range(8):
            #calculate G(u,v) as follows:
            G[u][v]=singleGUV(subimg,u,v,inv)
    return G

#Function for quantization of the DCT matrix
#http://stackoverflow.com/questions/29215879/how-can-i-generalize-the-quantization-matrix-in-jpeg-compression
def quantize_inv(G,quality=50):
    #defining the quantization matrix for a quality of 50% as per the JPEG standard
    s1=array([ 16,  11,  10,  16,  24,  40,  51,  61,  12,  12,  14,  19,  26,
        58,  60,  55,  14,  13,  16,  24,  40,  57,  69,  56,  14,  17,
        22,  29,  51,  87,  80,  62,  18,  22,  37,  56,  68, 109, 103,
        77,  24,  35,  55,  64,  81, 104, 113,  92,  49,  64,  78,  87,
       103, 121, 120, 101,  72,  92,  95,  98, 112, 100, 103,  99])

    if(quality<50):
        S=round(5000/float(quality))
    else:
        S=200-2*quality

    Q = floor((S*s1+50)/100).reshape(8,8).astype(np.int16)
    B = np.zeros((8,8),dtype='int16')
    for i in range(8):
        for j in range(8):
            B[i][j]=round(G[i][j]/float(Q[i][j]))*Q[i][j]
    return B.astype('int16')

#convert a PNG image into a "RAW" format for sending over UART
im=cv2.imread("/media/rohan/DISK_IMG/jpg.jpg")
#im=cv2.resize(im,(0,0),fx=0.5,fy=0.5)
#show(im)
rows1,cols1,ch = im.shape

f=open("original.dat","wb")
f.write(str(rows1)+" "+str(cols1)+"\n")
for i in range(rows1):
	for j in range(cols1):
		f.write(str(im[i][j][0])+" "+str(im[i][j][1])+" "+str(im[i][j][2]))
		f.write("\n")
f.write("\n")
f.close()

#Convert the colour space from RGB to YCrCb
# http://stackoverflow.com/questions/10443295/combine-3-separate-numpy-arrays-to-an-rgb-image-in-python
YCrCb=cv2.cvtColor(im,cv2.COLOR_RGB2YCR_CB)
Y=YCrCb[...,0]; Cr=YCrCb[...,1]; Cb=YCrCb[...,2];

#Pad the image such that it consists of only 16x16 blocks
#The Cr and Cb blocks are subsequently downsampled, so we have 4 8x8 Y blocks and two Cr,Cb 8x8 blocks
#Total of 6 8x8 blocks per 16x16 input block due to this downsampling operation
YCrCb=imgPad(YCrCb)
#show(YCrCb[rows/2:,:])
evalcos()       #evaluate the cosine matrix used for the DCT operation
'''
PSEUDOCODE

for each 16x16 block in the image:
    downsample the Cr and Cb components
    perform a 2D DCT on each of the six resulting 8x8 blocks
    quantize the resulting coefficients using a quantization matrix (higher weights to lower freq.)
    *zigzag encoding of the different coefficients, and Huffman+run-length encoding of the output
    *assemble the output using the JFIF format spec
'''
vint=vectorize(round)
#new set of rows,cols
rows,cols,ch = YCrCb.shape
print rows,cols,ch
for row in range(0,rows,8):
    for col in range(0,cols,8):
        #block = YCrCb[row:row+16,col:col+16]
        #Y = block[:,:,0]; Cb = block[:,:,1]; Cr = block[:,:,2]
        YCrCb[row:row+8,col:col+8,0] = vint(DCT_8x8_2D(quantize_inv(DCT_8x8_2D(YCrCb[row:row+8,col:col+8,0]),10),0,1)+128)
        YCrCb[row:row+8,col:col+8,1] = vint(DCT_8x8_2D(quantize_inv(DCT_8x8_2D(YCrCb[row:row+8,col:col+8,1]),10),0,1)+128)
        YCrCb[row:row+8,col:col+8,2] = vint(DCT_8x8_2D(quantize_inv(DCT_8x8_2D(YCrCb[row:row+8,col:col+8,2]),10),0,1)+128)
        #print row,col
        
im1=np.zeros((rows,cols,ch),dtype=np.uint8)
im1=cv2.cvtColor(YCrCb,cv2.COLOR_YCR_CB2RGB)
show(im1[:rows1,:cols1])
t2=time.time()
print "done",t2-t1

f=open("compressed.dat","wb")
f.write(str(rows1)+" "+str(cols1)+"\n")
for i in range(rows1):
	for j in range(cols1):
		f.write(str(im1[i][j][0])+" "+str(im1[i][j][1])+" "+str(im1[i][j][2]))
		f.write("\n")
f.write("\n")
f.close()
