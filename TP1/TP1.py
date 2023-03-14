### Image Processing - Term Project 1
### 2016121150 윤준영
### 2022/05/01

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import time





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

img = cv2.imread('lena.bmp')
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
h = histogram(gray,plot=1)







# ============================== 1. Image denoising ============================== 
print('\n1. Image denoising')
# 1.1), 1.2) Apply Noise and compute PSNR
print('1.2. Compute PSNR')

def random_noise(size, par=1, type='uniform'):
    ### random noise generation ###
    # input : noise size (MxN), parameter, noise type
    # output : noise (MxN array)

    # type : 'gaussian', 'uniform', 'impulse'
    # Gaussian / par : variance
    # Uniform / par : max value of noise
    # Impulse / par : p% (0~100)
    if type == 'gaussian':
        noise = np.random.normal(0, np.sqrt(par), size)
    else:
        noise = np.random.uniform(-1, 1, size)
        if type == 'impulse':
            inoise = np.zeros(noise.shape)
            par = 1 - par/100
            inoise[noise>par] = 255
            inoise[noise<-par] = -255
            noise = inoise
        else: noise = noise * par
    return noise

def apply_noise(img, noise):
    ### apply noise to image ###
    # input : image (MxN array), noise (MxN array)
    # output : noisy image (MxN array)
    img = img.astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img

def error(img1, img2, type='PSNR'):
    ### error calculation ###
    # input : 2 images (MxN array)
    # output : error (MSE or PSNR)
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    error = np.mean((img1 - img2)**2)
    if type=='PSNR':
        error = 10 * math.log10(255**2/error)
    return error


h=list(); n=0; pl=0;
par = [[50, 100, 200],[1,3,5],[10,30,50]]
pt = ['var', 'p(%)', 'max']
for noise_type in ['gaussian', 'impulse', 'uniform']:
    plt.figure(f'1.1 Noise generation {noise_type}', figsize=(13, 4))
    plt.suptitle(f'Noise generation {noise_type}')
    for p in par[pl]:
        plt.subplot(1,3,n+1); plt.title(f'{pt[pl]}={p}');
        h.append(histogram(random_noise(gray.shape, p, noise_type),size=500, plot=1))
        n+=1
    plt.tight_layout()
    #plt.savefig(f'1.1) Noise generation {noise_type}.png')
    n=0
    pl+=1
plt.show()

gray_n = list(); n=0; pl=0;
plt.figure('1.1 Apply noise', figsize=(9, 9))
plt.suptitle('Apply noise')
par = [[50, 100, 200],[1,3,5],[10,30,50]]
pt = ['var', 'p(%)', 'max']
for noise_type in ['gaussian', 'impulse', 'uniform']:
    for p in par[pl]:
        gray_n.append(apply_noise(gray, random_noise(gray.shape, p, noise_type)))
        plt.subplot(3,3,n+1); plt.title(f'{noise_type} noise {pt[pl]}={p}'); plt.axis('off')
        plt.imshow(gray_n[n], cmap='gray', vmin=0, vmax=255)
        print(f'{noise_type} noise {pt[pl]}={p}, PSNR : {error(gray, gray_n[n]):.2f}dB')
        n+=1
    pl+=1
plt.tight_layout()
#plt.savefig('1.1 Apply noise.png')
plt.show()

# 1.3) Implement denoising filters.

def filter_gen(par1, par2=1, type='gaussian', norm=1):
    ### filter generation ###
    # input : par1, par2, filter type, normalization
    # output : filter
    if type=='gaussian':
        # gaussian filter
        # par1 : size, par2 : sigma
        ar = np.arange(-par1//2+1,par1//2+1)
        [y, x] = np.meshgrid(ar,ar)
        fil = np.exp(-(x**2+y**2)/(2*par2**2))
    if type=='log':
        # LoG filter
        # par1 : size, par2 : sigma
        ar = np.arange(-par1//2+1,par1//2+1)
        [y, x] = np.meshgrid(ar,ar)
        t = -(y**2+x**2)/(2*par2**2)
        fil = -1/(np.pi*par2**4)*(1+t)*np.exp(t)
        norm = 0
    elif type=='box':
        # box filter
        # par1 : size, par2 : 1 x par1^2 array
        fil = par2.reshape(par1,par1)
    elif type=='average':
        # average filter
        # par1 : size
        fil = np.ones((par1,par1))
    elif type=='bilinear':
        # bilinear interpolation filter
        # size = (2 x par1 - 1) x (2 x par2 - 1)
        ar1 = np.arange(1, par1)/par1
        ar1 = np.hstack((ar1,(1), np.flip(ar1)))
        ar2 = np.arange(1, par2)/par2
        ar2 = np.hstack((ar2,(1), np.flip(ar2)))
        [y, x] = np.meshgrid(ar2,ar1)
        fil = x*y
        norm = 0
    if norm==1 and np.sum(fil)!=0 : fil = fil/np.sum(fil)
    return fil

def apply_filter(img, fil):
    ### appling filter to image ###
    # input : image (MxN array), filter (mxn array)
    # output : filtered image (MxN array)
    result = np.copy(img)
    h = fil.shape[0]//2
    for y in range(h, img.shape[0]-h):
        for x in range(h, img.shape[1]-h):
            sub_img = img[y-h:y+h+1, x-h:x+h+1]
            result[y,x] = np.sum(fil * sub_img).astype(np.uint8)
    return result

def apply_filter2(img, fil, i=1, ab=1):
    ### appling filter to image (Matrix calculate) ###
    # input : image (MxN array), filter (mxn array),
    #         i=1:output to integer, ab=1:output to absolute value
    # output : filtered image (MxN array)
    h1 = fil.shape[0]//2
    h2 = fil.shape[1]//2
    result = np.zeros((img.shape[0]+2*h1, img.shape[1]+2*h2))
    ar1 = np.arange(-h1,h1+1)
    ar2 = np.arange(-h2,h2+1)
    for y in ar1:
        for x in ar2:
            sub = np.pad(img,((h1-y,h1+y),(h2-x,h2+x)),constant_values=0)
            result += fil[y+h1,x+h2] * sub
    if ab==1 : result = abs(result)
    if i==1 : result = result.astype(np.uint8)
    result = result[h1:-h1,h2:-h2]
    return result

print('1.3.0) Faster filtering')
fil = filter_gen(7)
s1 = time.time()
gray_f1 = apply_filter(gray_n[1], fil)
s2 = time.time()
gray_f2 = apply_filter2(gray_n[1], fil)
s3 = time.time()
plt.figure('1.3.0 faster filtering', figsize=(9,5));
plt.subplot(1,2,1); plt.title('function 1'); plt.axis('off')
plt.imshow(gray_f1, cmap='gray', vmin=0, vmax=255)
plt.subplot(1,2,2); plt.title('function 2'); plt.axis('off')
plt.imshow(gray_f2, cmap='gray', vmin=0, vmax=255)
plt.tight_layout()
print(f'apply_filter : {(s2-s1):.3f}sec')
print(f'apply_filter2 : {(s3-s2):.3f}sec')
print(f'difference - PSNR : {error(gray_f1, gray_f2):.2f}dB')
#plt.savefig('1.3.0) Faster filtering.png')
plt.show()

# 1.3.a) Apply Gaussian filter
print('1.3.a) Gaussian filter')
gray_f = list(); n=0;
plt.figure('1.3.a gaussian filter', figsize=(12, 5))
plt.suptitle('Gaussian filter')
sigma = [0.5, 1, 3]; size = [3, 7, 19]
for n in range(3):
    gray_f.append(apply_filter2(gray_n[2], filter_gen(size[n],sigma[n])))
    plt.subplot(1,3,n+1); plt.title(f'$\sigma$={sigma[n]}({size[n]}x{size[n]})'); plt.axis('off')
    plt.imshow(gray_f[n], cmap='gray', vmin=0, vmax=255)
    print(f'gaussian filter, sigma={sigma[n]}({size[n]}x{size[n]}), PSNR : {error(gray, gray_f[n]):.2f}dB')
plt.tight_layout()
#plt.savefig('1.3.a) Gaussian filter.png')
plt.show()

# 1.3.b) Apply bilateral filter
def bilateral_filter(img, ss, sr):
    ### bilateral filtering ###
    # input : image (MxN array), sigma_s : spatial std dev, sigma_r : range std dev
    hs = int((ss*6//2*2+1)/2)
    hr = int((sr*6//2*2+1)/2)
    h = max(hs,hr)
    hd = hs - hr
    img = np.pad(img, h, constant_values=0)
    result = np.zeros(img.shape)
    spa = filter_gen(hs*2+1, ss, norm=0)
    if hr>hs :
        spa = np.pad(spa, hr-hs, constant_values=0)
        hd = 0
    for i in range(h, img.shape[0]-h):
        for j in range(h, img.shape[1]-h):
            sub = img[i-hr:i+hr+1, j-hr:j+hr+1].astype(np.float32)
            ran = np.exp(-((sub-sub[hr,hr])/255)**2/(2*sr**2))
            ran = np.pad(ran, hd, constant_values=0)
            sub = np.pad(sub, hd, constant_values=0)
            fil = spa*ran;
            fil = fil/np.sum(fil)
            result[i,j]=np.sum(fil*sub)
    result = result[h:-h,h:-h].astype(np.uint8)
    return result

print('1.3.b) Bilateral filter')
print('\n Please wait... \n')
gray_bf = list(); n=0;
plt.figure('1.3.b Bilateral filter', figsize=(9, 9))
plt.suptitle('Bilateral filter')
ss = 1; sr = [0.4,0.7,1]; sigma = 1;
plt.subplot(2,2,4); plt.title(f'gaussian filter $\sigma$={sigma}'); plt.axis('off')
plt.imshow(gray_f[1], cmap='gray', vmin=0, vmax=255)
for n in range(3):
    s1 = time.time()
    gray_bf.append(bilateral_filter(gray_n[2],ss,sr[n]))
    s2 = time.time()
    plt.subplot(2,2,n+1); plt.title(f'bilateral $\sigma_s,\sigma_r$={ss,sr[n]}'); plt.axis('off')
    plt.imshow(gray_bf[n], cmap='gray', vmin=0, vmax=255)
    print(f'bilateral filter, sigma_r,sigma_s={ss,sr[n]}, PSNR : {error(gray, gray_bf[n]):.2f}dB, {(s2-s1):.3f}sec')
plt.tight_layout()
#plt.savefig('1.3.b) Bilateral filter.png')
plt.show()

# 1.3.c) Apply average filter

print('1.3.c) Average filter')
gray_af = list(); n=0;
plt.figure('1.3.c average filter', figsize=(12, 5))
plt.suptitle('Average filter')
size = [3, 5, 7]
for n in range(3):
    gray_af.append(apply_filter2(gray_n[2], filter_gen(size[n],type='average')))
    plt.subplot(1,3,n+1); plt.title(f'filter size({size[n]}x{size[n]})'); plt.axis('off')
    plt.imshow(gray_af[n], cmap='gray', vmin=0, vmax=255)
    print(f'average filter({size[n]}x{size[n]}), PSNR : {error(gray, gray_af[n]):.2f}dB')
plt.tight_layout()
#plt.savefig('1.3.c) Average filter.png')
plt.show()

# 1.4) Compare result of subjective and objective measurements








# ============================== 2. Image demosaicking ============================== 
print('\n2. Image demosaicking')
# 2.1) Design an algorithm to implement CFA patterns

def mosaick(rgb, cfa):
    # intput : rgb image (MxNx3 array), cfa array #
    # output : mosaicked image
    mo = np.zeros(rgb.shape,dtype=np.uint8)
    [y, x] = cfa.shape
    for i in range(y):
        for j in range(x):
            mo[i::y, j::x, cfa[i,j]] = rgb[i::y, j::x, cfa[i,j]]
    return mo

cfa_a = np.array([[1,0],[2,1]])
cfa_b = np.array([[1,0,1,2],[1,2,1,0]])
cfa_c = np.array([[1,0],[1,2],[0,1],[2,1]])
cfa_d = np.array([[1,0,2],[1,0,2]])
cfa_e = np.array([[0,2,1],[1,0,2]])
cfa_f = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,1,0]])
plt.figure('2.1 CFA', figsize=(12, 8))
plt.suptitle('CFA patterns')
num = ['a','b','c','d','e','f']; n=0; rgb_mo = list()
for i in [cfa_a, cfa_b, cfa_c, cfa_d, cfa_e, cfa_f]:
    rgb_mo.append(mosaick(rgb, i))
    plt.subplot(2,3,n+1); plt.title(f'({num[n]})'); plt.axis('off')
    plt.imshow(rgb_mo[n], vmin=0, vmax=255)
    n+=1
plt.tight_layout()
#plt.savefig('2.1 CFA patterns.png')
plt.show()

# 2.2), 2.3) Build demosaicking algorithm and Compute PSNR

def demosaick(mo, cfa):
    [y, x] = cfa.shape
    cfa_ = np.zeros((y,x,3))
    cfa_[:,:,0][cfa==0] = 1
    cfa_[:,:,1][cfa==1] = 1
    cfa_[:,:,2][cfa==2] = 1
    cfa_ = np.vstack((cfa_, cfa_))
    cfa_ = np.hstack((cfa_, cfa_))
    result = np.zeros(mo.shape)
    fil = filter_gen(y, x, type='bilinear')
    [y, x] = fil.shape
    result[:,:,0] = apply_filter2(mo[:,:,0], fil/np.sum(cfa_[:y,:x,0]*fil))
    result[:,:,1] = apply_filter2(mo[:,:,1], fil/np.sum(cfa_[:y,:x,1]*fil))
    result[:,:,2] = apply_filter2(mo[:,:,2], fil/np.sum(cfa_[:y,:x,2]*fil))
    return result.astype(np.uint8)

print('2.3) Image demosaicking PSNR')
plt.figure('2.2 Demosaick', figsize=(12, 8))
plt.suptitle('Demosaick')
num = ['a','b','c','d','e','f']; n=0; rgb_dm=list()
for i in [cfa_a, cfa_b, cfa_c, cfa_d, cfa_e, cfa_f]:
    rgb_dm.append(demosaick(rgb_mo[n], i))
    plt.subplot(2,3,n+1); plt.title(f'({num[n]})'); plt.axis('off')
    plt.imshow(rgb_dm[n], vmin=0, vmax=255)
    print(f'Demosaick ({num[n]}), PSNR : {error(rgb, rgb_dm[n]):.2f}dB')
    n+=1
plt.tight_layout()
#plt.savefig('2.2 Demosaick.png')
plt.show()








# ============================== 3. Local Descriptor; SIFT ============================== 
print('\n3. SIFT')
# 3.1) Implement the image pyramid generation function

def downscale(img, n):
    ### make 1/n image ###
    result = np.zeros((img.shape[0]//n, img.shape[1]//n))
    img = img[:result.shape[0]*n, :result.shape[1]*n]
    d = 0
    for i in range(n):
        for j in range(n):
            result += img[i::n,j::n]
            d = d+1
    result = result/d
    result = result.astype(np.uint8)
    return result

def bilinear_scale(img, n):
    ### make n x image (bilinear scaling) ###
    result = np.zeros((img.shape[0]*n, img.shape[1]*n))
    result[::n,::n] = img
    fil = filter_gen(n, n, type='bilinear')
    result = apply_filter2(result, fil)
    return result

def sift_pyramid_gen(img, sigma=1.6, k=6, o=3):
    ### SIFT pyramid generation ###
    # input : img : M x N image, th : threshold, sigma : scaling factor   
    #         k : number of images, o : number of octave
    # output : pyramid(list of octaves), ( ex) pyramid[0] = 0 octave(M x N x k) )
    pyramid = list()
    for i in range(o):
        mx = 0
        img_o = downscale(img, 2**i)
        img_f = np.zeros([img_o.shape[0], img_o.shape[1], k])
        for n in range(k):
            if n==0 : sig = np.sqrt(sigma**2 - 0.5**2)
            else : sig = np.sqrt((sigma*2**(1/3*n))**2 - (sigma*2**(1/3*(n-1)))**2)
            size = sig*6//2*2+1
            fil = filter_gen(size, sig)
            if n==0 : img_f[:,:,n] = apply_filter2(img_o, fil, i=0)
            else : img_f[:,:,n] = apply_filter2(img_f[:,:,n-1], fil, i=0)
        pyramid.append(img_f)
    return pyramid

def plot_pyramid(pyramid, dog=0):
    ### plotting pyramid ###
    # input : pyramid(list of octaves)
    # output : -
    k = pyramid[0].shape[2]
    for o in range(len(pyramid)):
        plt.figure(f'Octave %d'%o, figsize=(12, 3))
        for i in range(k):
            plt.subplot(1,k,i+1)
            sigma = 1.6*2**(1/3*i)
            if dog!=1 :
                plt.imshow(pyramid[0][:,:,i], cmap='gray', vmin=0, vmax=255)
                plt.title(f'$\sigma$={sigma:.4f}')
            else : plt.imshow(pyramid[0][:,:,i], cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Octave {o}, size={pyramid[o].shape[0:2]}')
        plt.tight_layout()
        #if dog==0 : plt.savefig(f'3.1 Octave {o}.png')
        #else : plt.savefig(f'3.1 DOG Octav {o}.png')
    plt.show()
    
def sift_kp(pyramid, th):
    ### keypoint detection from sift pyramid ###
    # input : pyramid(list of octaves), threshold
    # output : keypoint(y, x, o, i)
    mx = [5, 7, 9, 11, 15, 19]
    kp = np.array([[0, 0, 0, 0]])
    dog = list()
    for o in range(len(pyramid)):
        gaussian = pyramid[o]
        dog.append(gaussian[:,:,:-1] - gaussian[:,:,1:])
        #kp = np.array([[0, 0, 0, 0, 0]])
        for n in range(1,4):
            [y, x] = dog[o].shape[0:2]
            off = mx[n]
            dogn = dog[o][off:y-off,off:x-off,[n,n-1,n+1]]
            for i in [0, -1, 1]:
                for j in [0, -1, 1]:
                    dogn = np.append(dogn, dog[o][off+i:y-off+i,off+j:x-off+j,[n,n-1,n+1]], axis=2)
            dogn = dogn[:,:,3:]
            dogn = abs(dogn)
            dogn = dogn/np.max(dogn)
            pos = np.where((np.max(dogn[:,:,1:], axis=2)!=np.max(dogn, axis=2))*(dogn[:,:,0]>th))
            kp_n = np.transpose([(pos[0]+off)*2**o, (pos[1]+off)*2**o])
            kp_n = np.pad(kp_n, ((0,0),(0,1)), constant_values=o)
            kp_n = np.pad(kp_n, ((0,0),(0,1)), constant_values=i)
            kp = np.append(kp, kp_n, axis=0)
    return kp[1:,:], dog

gray_py = sift_pyramid_gen(gray)
plot_pyramid(gray_py)
gray_kp, gray_dog = sift_kp(gray_py, 0.5)
plot_pyramid(gray_dog, dog=1)

def rot90(img):
    R=len(img[:,0])
    C=len(img[0,:])
    image_new = np.zeros(img.shape)
    for r in range(R):
      for c in range(C):
        image_new[c,R-1-r]=img[r,c]
    return image_new

gray_r = rot90(gray)
gray_r = bilinear_scale(gray_r, 2)
gray_r_py = sift_pyramid_gen(gray_r)
#plot_pyramid(gray_r_py)
gray_r_kp, gray_r_dog = sift_kp(gray_r_py, 0.5)
#plot_pyramid(gray_r_dog, dog=1)


# 3.2) Verify that response of DoG and LoG is similar

def log_pyramid_gen(img, sigma=1, k=6, o=3):
    ### LoG pyramid generation ###
    # input : img : M x N image, th : threshold, sigma : scaling factor   
    #         k : number of images, o : number of octave
    # output : pyramid(list of octaves), ( ex) pyramid[0] = 0 octave(M x N x k) )
    pyramid = list()
    for i in range(o):
        mx = 0
        img_o = downscale(img, 2**i)
        img_f = np.zeros([img_o.shape[0], img_o.shape[1], k])
        for n in range(k):
            if n==0 : sig = np.sqrt(sigma**2 - 0.5**2)
            else : sig = np.sqrt((sigma*2**(1/3*n))**2 - (sigma*2**(1/3*(n-1)))**2)
            size = sig*6//2*2+1
            fil = filter_gen(size, sig, type='log')
            img_f[:,:,n] = apply_filter2(img_o, fil, i=0, ab=0)
        pyramid.append(img_f)
    return pyramid


# 3.3) Using generated image pyramid, detect key points, check validity
plt.figure('SIFT')
plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
plt.plot(gray_kp[:,1], gray_kp[:,0], 'ro', ms=1.5)
plt.title('SIFT')
plt.tight_layout()
#plt.savefig('3.3 SIFT - keypoints.png')

plt.figure('SIFT - OpenCV')
sift = cv2.SIFT_create()
kp = sift.detect(gray,None)
cv_img = cv2.drawKeypoints(gray,kp,img)
plt.imshow(cv_img)
plt.title('SIFT - OpenCV Library')
plt.tight_layout()
#plt.savefig('3.3 SIFT - keypoints(OpenCV).png')


plt.figure('SIFT(90deg, 2xscaled)')
plt.imshow(gray_r, cmap='gray', vmin=0, vmax=255)
plt.plot(gray_r_kp[:,1], gray_r_kp[:,0], 'ro', ms=1.5)
plt.title('SIFT(90deg, 2xscaled)')
plt.tight_layout()
#plt.savefig('3.3 SIFT - keypoints(90deg, 2xscaled).png')

plt.figure('SIFT - OpenCV, 90deg, 2xscaled')
sift_r = cv2.SIFT_create()
kp_r = sift_r.detect(gray_r,None)
cv_r_img = cv2.drawKeypoints(gray_r,kp_r,gray_r)
plt.imshow(cv_r_img)
plt.title('SIFT - OpenCV, 90deg, 2xscaled')
plt.tight_layout()
#plt.savefig('3.3 SIFT - keypoints(OpenCV,90deg, 2xscaled).png')

plt.show()

