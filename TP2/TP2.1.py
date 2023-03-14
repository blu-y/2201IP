### Image Processing - Term Project 2.1
### 2016121150 윤준영
### 2022/06/19
# Python ver. 3.10.4

import numpy as np
import matplotlib.pyplot as plt
import math
from random import seed
from random import sample
import cv2
import time
seed(10)




# ============================== 0. Import & convert image ============================== 

def B2R(img):
    ### BGR to RGB ###
    # input : BGR image (MxNx3 array)
    # output : RGB image (MxNx3 array)
    rgb = np.copy(img)
    rgb[:,:,0] = img[:,:,2]
    rgb[:,:,2] = img[:,:,0]
    return rgb

def B2G(img):
    ### BGR to Grayscale ###
    # input : BGR image (MxNx3 array)
    # output : grayscaled image (MxN array)
    img = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]
    return img.astype(np.uint8)

def R2Y(rgb):
    ### RGB to YUV ###
    # input : RGB image (MxNx3 array)
    # output : YUV image (MxNx3 array)
    yuv = np.zeros_like(rgb).astype(np.int16)
    yuv[:,:,0] = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    yuv[:,:,1] = - 0.147*rgb[:,:,0] - 0.289*rgb[:,:,1] + 0.436*rgb[:,:,2]
    yuv[:,:,2] = 0.615*rgb[:,:,0] - 0.515*rgb[:,:,1] - 0.100*rgb[:,:,2]
    return yuv

def Y2R(yuv):
    ### YUV to RGB ###
    # input : YUV image (MxNx3 array)
    # output : RGB image (MxNx3 array)
    rgb = np.zeros_like(yuv)
    rgb[:,:,0] = yuv[:,:,0] + 1.140*yuv[:,:,2]
    rgb[:,:,1] = yuv[:,:,0] - 0.395*yuv[:,:,1] - 0.581*yuv[:,:,2]
    rgb[:,:,2] = yuv[:,:,0] + 2.032*yuv[:,:,1]
    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)

img = cv2.imread('house.bmp')
rgb = B2R(img)
gray = B2G(img)
plt.figure('0. Original image', figsize=(9, 5))
plt.subplot(121); plt.title('rgb'); plt.axis('off')
plt.imshow(rgb, vmin=0, vmax=255)
plt.subplot(122); plt.title('grayscale'); plt.axis('off')
plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
plt.suptitle('Original image')
plt.tight_layout()
#plt.savefig('0. Original image.png')
plt.show()

def histogram(img, size=1, bit=8, plot=0):
## img : gray image, bit : bit
    h = np.zeros(2**bit)
    img = np.rint(img)
    if size!=1:
        h = np.zeros(size)
        ar = np.linspace(np.min(img)-1, np.max(img)+1,size)
        for p in range(size-1):
            h[p] = np.sum((ar[p]<img)*(img<ar[p+1]))
        if plot == 1:
            plt.bar(ar,h)
    else :
        ar = range(2**bit)
        for p in ar:
            h[p] = np.count_nonzero(img==p)
        if plot == 1:
            plt.figure('hisogram', figsize=(4, 4))
            plt.bar(np.arange(2**bit),h)
            plt.title('histogram')
            plt.tight_layout()
            #plt.savefig(f'0.1) Histogram.png')
            plt.show()
    return h
h = histogram(gray,plot=0)







# ============================== 1. K-means/Mean shift ============================== 
print('\n1. K-means/Mean shift')
# 1.1) Implement the k-means and Mean shift algorithms.
def dist(x,y):
### Distance ###
    d = np.sum((x-y)**2)
    return d

def distmat(x,y):
### 3D distance, Matrix ver. ###
    d = np.sum((x-y)**2,axis=2)
    return d

def gk(x):
### Gaussian kernel ###
### x : vector
  norm = np.linalg.norm(x)
  if norm <= 1 : return np.exp(-norm**2) 
  else : return 0

def gk_map(x):
### Gaussian kernel ###
### x : vector
    norm = np.linalg.norm(x,axis=1)
    gk = np.zeros_like(norm)
    gk[norm<=1] = np.exp(-norm**2)[norm<=1]
    return gk

# 1.1.a) K-means
def kmeans_image(rgb, k, maxiter=50, plot=0, color='rgb', pos='x'):
### K-Means Algorithm ###
### input - rgb : rgb image, k : int
### output - data label (0 ~ k-1), labeled image
    # initialize old and new index arrays > ind, nind
    # initialize distance map > dis
    # initialize grouping point array > g
    if color == 'yuv' : rgb = R2Y(rgb)
    if pos != 'x' :
        yp, xp = np.meshgrid(np.arange(rgb.shape[0]), np.arange(rgb.shape[1]))
        rgb = np.dstack((rgb, np.transpose(yp)/pos, np.transpose(xp)/pos))
        pos = f'{1/pos}'
    ind = np.zeros_like(rgb[:,:,0])
    nind = np.copy(ind)
    dis = np.zeros(ind.shape + (k,))
    g = np.zeros([k,rgb.shape[2]])
    # generate random k points
    rany = sample(range(rgb.shape[0]), k)
    ranx = sample(range(rgb.shape[1]), k)
    # set initial grouping point and calculate distance map (y, x, k)
    for i in range(k):
        g[i,:] = rgb[rany[i], ranx[i], :]
        dis[:,:,i] = distmat(rgb, g[i])
    # initial grouping with distance, save new index > nind
    nind = np.argmin(dis, axis=2)
    iter = 0;
    # repeat until old and new index array is same
    while(not(np.array_equal(nind,ind))):
        # make new index to old, group initialize
        ind = np.copy(nind)
        # make new grouping point with grouped data
        for i in range(k):
            if rgb[ind==i].size != 0: g[i,:] = np.average(rgb[ind==i],axis=0)
        # grouping with distance, save new index > nind
        for i in range(k):
            dis[:,:,i] = distmat(rgb, g[i])
        nind = np.argmin(dis, axis=2)
        iter = iter + 1
        if iter > maxiter:
            print("max iteration")
            nind = ind
            break
    sort = np.zeros_like(rgb)
    g = g.astype(np.int16)
    for i in range(k):
        sort[ind==i] = g[i]
    g = g.reshape([k,1,rgb.shape[2]])
    ## plot
    if color == 'yuv':
        sort = Y2R(sort)
        g = Y2R(g)
    if pos != 'x':
        sort = sort[:,:,:3].astype(np.uint8)
        g = g[:,:,:3].astype(np.uint8)
    if plot == 1:
        plt.figure(f'K-means, k = {k}, color = {color}, pos = {pos}')
        plt.subplot(1,2,1); plt.title('sorted'); plt.axis('off')
        plt.imshow(sort)
        plt.subplot(1,2,2); plt.title('label'); plt.axis('off')
        plt.imshow(g)
        plt.tight_layout()
        #plt.savefig(f'K-means, k = {k}, color = {color}, pos = {pos}.png')
        #plt.show()
    return nind, sort

# 1.1.b) Mean shift
def downscale(img, n):
    ### make 1/n image ###
    result = np.zeros((img.shape[0]//n, img.shape[1]//n, img.shape[2]))
    img = img[:result.shape[0]*n, :result.shape[1]*n]
    d = 0
    for i in range(n):
        for j in range(n):
            result += img[i::n,j::n]
            d = d+1
    result = result/d
    return result.astype(np.int16)

def meanshift_image(rgb, h=50, s=4, plot=0, color='rgb', pos='x'):
### Mean shift Algorithm ###
### input - rgb : rgb image, h : kernel parameter
### output - data label (0 ~ k-1), labeled image
    # n : no of samples, y : current shifting point
    # v : convergence point, E = epsilon, c : cluster, c_center : center of cluster
    if color == 'yuv' : rgb = R2Y(rgb)
    if pos != 'x' :
        yp, xp = np.meshgrid(np.arange(rgb.shape[0]), np.arange(rgb.shape[1]))
        rgb = np.dstack((rgb, np.transpose(yp)/pos, np.transpose(xp)/pos))
        pos = f'{1/pos}'
    rgb_ds = downscale(rgb, s)
    n = rgb_ds[:,:,0].size
    x = rgb_ds.reshape([n, rgb_ds.shape[2]]).astype(np.int16)
    v = np.zeros_like(x)
    c = np.zeros(n)
    c_new = np.copy(c)
    c_center = []
    E = h/10
    ## find convergence point v with downscaled image
    for i in range(n):
        y = np.copy(x[i])
        e = E+1
        # calculate y[t+1] from y[t] and x[i]
        while(e>E):
            s = (x-y)/h
            gk = gk_map(s).reshape([n,1])
            myu = np.sum(gk * (x-y), axis=0)
            myl = np.sum(gk)
            if myl == 0 : y_new = y
            else : y_new = y + myu/myl
            e = dist(y,y_new)
            y = y_new
        # save convergence point to v
        v[i] = y
    ## grouping v
    i = 0
    while(1):
        # pick random point of i cluster
        rest = np.asarray(np.nonzero(c==i))[0,:].tolist()
        ran = sample(rest,1)
        c_center.append(v[ran])
        # regroup i cluster to i and i+1
        for j in rest:
            if dist(v[j],c_center[i]) < h : c_new[j] = i
            else : c_new[j] = i+1
        # if there's no i+1 group; break
        if (c==c_new).min() :
            i=i+1
            break
        c_center[i] = x[c_new==i].mean(axis=0)
        c = np.copy(c_new)
        i = i+1
    ## grouping image with c_center
    sort = np.zeros_like(rgb)
    dis = np.zeros(rgb[:,:,0].shape + (i,))
    for k in range(i):
        dis[:,:,k] = distmat(rgb, c_center[k])
    ind = np.argmin(dis,axis=2)
    for k in range(i):
        sort[ind==k] = c_center[k]
    c_center = np.vstack(c_center).reshape([i,1,rgb.shape[2]]).astype(np.int32)
    if color == 'yuv' :
        sort = Y2R(sort)
        c_center = Y2R(c_center)
    ## plot 
    if pos != 'x':
        sort = sort[:,:,:3].astype(np.uint8)
        c_center = c_center[:,:,:3].astype(np.uint8)
    if plot == 1:
        plt.figure(f'Mean shift, h = {h}, label = 0~{i-1}, color = {color}, pos = {pos}')
        plt.subplot(1,2,1); plt.title('sorted'); plt.axis('off')
        plt.imshow(sort[:,:,:3])
        plt.subplot(1,2,2); plt.title('label'); plt.axis('off')
        plt.imshow(c_center)
        plt.tight_layout()
        #plt.savefig(f'Mean shift, h = {h}, label = 0~{i-1}, color = {color}, pos = {pos}.png')
        #plt.show()
    return ind, sort

# 1.2) Cluster image based on (R, G, B)
kmeans_image(rgb, 3, plot=1)
kmeans_image(rgb, 5, plot=1)
#kmeans_image(rgb, 8, maxiter=100, plot=1)
meanshift_image(rgb, h=40, s=8, plot=1)
meanshift_image(rgb, h=50, s=8, plot=1)
#meanshift_image(rgb, h=60, s=8, plot=1)
plt.show()

# 1.3) Convert RGB to YUV and cluster
kmeans_image(rgb, 3, plot=1, color='yuv')
kmeans_image(rgb, 5, plot=1, color='yuv')
#kmeans_image(rgb, 8, maxiter=100, plot=1, color='yuv')
meanshift_image(rgb, h=30, s=8, plot=1, color='yuv')
meanshift_image(rgb, h=35, s=8, plot=1, color='yuv')
#meanshift_image(rgb, h=40, s=8, plot=1, color='yuv')
plt.show()

# 1.4) Use position information in addition to the (R, G, B) or (Y, U, V)
kmeans_image(rgb, 3, plot=1, pos=2)
kmeans_image(rgb, 5, plot=1, pos=2)
#kmeans_image(rgb, 8, maxiter=100, plot=1, pos=2)
meanshift_image(rgb, h=50, s=8, plot=1, pos=2)
meanshift_image(rgb, h=65, s=8, plot=1, pos=2)
#meanshift_image(rgb, h=80, s=8, plot=1, pos=2)
plt.show()

kmeans_image(rgb, 3, plot=1, color='yuv', pos=2)
kmeans_image(rgb, 5, plot=1, color='yuv', pos=2)
#kmeans_image(rgb, 8, maxiter=100, plot=1, color='yuv', pos=2)
meanshift_image(rgb, h=50, s=8, plot=1, color='yuv', pos=2)
meanshift_image(rgb, h=65, s=8, plot=1, color='yuv', pos=2)
#meanshift_image(rgb, h=80, s=8, plot=1, color='yuv', pos=2)
plt.show()

# 1.5) Repeat the image segmentation for cartoon or game image
img = cv2.imread('FN1.bmp')
rgb = B2R(img)
kmeans_image(rgb, 3, plot=1)
kmeans_image(rgb, 5, plot=1)
#kmeans_image(rgb, 8, maxiter=100, plot=1, color='yuv')
meanshift_image(rgb, h=60, s=32, plot=1)
meanshift_image(rgb, h=70, s=32, plot=1)
#meanshift_image(rgb, h=80, s=32, plot=1)
plt.show()

kmeans_image(rgb, 3, plot=1, color='yuv')
kmeans_image(rgb, 5, plot=1, color='yuv')
#kmeans_image(rgb, 8, maxiter=100, plot=1, color='yuv')
meanshift_image(rgb, h=40, s=32, plot=1, color='yuv')
meanshift_image(rgb, h=50, s=32, plot=1, color='yuv')
#meanshift_image(rgb, h=60, s=32, plot=1, color='yuv')
plt.show()

kmeans_image(rgb, 3, plot=1, pos=2)
kmeans_image(rgb, 5, plot=1, pos=2)
#kmeans_image(rgb, 8, maxiter=100, plot=1, pos=2)
meanshift_image(rgb, h=170, s=32, plot=1, pos=2)
meanshift_image(rgb, h=200, s=32, plot=1, pos=2)
#meanshift_image(rgb, h=230, s=32, plot=1, pos=2)
plt.show()

kmeans_image(rgb, 3, plot=1, color='yuv', pos=2)
kmeans_image(rgb, 5, plot=1, color='yuv', pos=2)
#kmeans_image(rgb, 8, maxiter=100, plot=1, color='yuv', pos=2)
meanshift_image(rgb, h=150, s=32, plot=1, color='yuv', pos=2)
meanshift_image(rgb, h=170, s=32, plot=1, color='yuv', pos=2)
#meanshift_image(rgb, h=200, s=32, plot=1, color='yuv', pos=2)
plt.show()
