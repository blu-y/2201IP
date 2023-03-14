### Image Processing - Term Project 2.2
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
h = histogram(gray)

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
            sub = np.pad(img,((h1-y,h1+y),(h2-x,h2+x)),'edge')
            result += fil[y+h1,x+h2] * sub
    if ab==1 : result = abs(result)
    if i==1 : result = result.astype(np.uint8)
    result = result[h1:-h1,h2:-h2]
    return result

def binary_image(img, th, v):
    binary = np.zeros_like(img)
    binary[img>=th] = v
    return binary



# ============================== 2. Edge detector using ML ============================== 
print('\n2. Edge detector using ML')
# 2.1) Use a built-in canny edge detector to obtain edge images.
edge = cv2.Canny(gray, 100, 150)
plt.figure('Ground truth, opencv canny edge'); plt.axis('off')
plt.imshow(edge, cmap='gray')
plt.suptitle('Ground truth(opencv_canny)')
plt.tight_layout()
#plt.savefig('2.1 Ground truth(opencv_canny).png')
plt.show()

# 2.2) Apply Prewitt and Sobel filters to check the performance.
def edge_filter(img, fil, th=30, plot=0):
    if fil=='prewitt':
        fil_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])/6
        fil_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])/6
    if fil=='sobel':
        fil_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])/8
        fil_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/8
    edge_y = apply_filter2(gray, fil_y)
    edge_x = apply_filter2(gray, fil_x)
    edge = edge_y + edge_x
    edge = binary_image(edge, th, 255)
    if plot==1:
        plt.figure(f'Edge_filter({fil})')
        plt.title(f'Edge_filter({fil})'); plt.axis('off')
        plt.imshow(edge, cmap='gray')
        plt.tight_layout()
        #plt.savefig(f'2.2 Edge_filter({fil}).png')
        #plt.show()
    return edge

edge_prewitt = edge_filter(gray, fil='prewitt', th=20, plot=1)
edge_sobel = edge_filter(gray, fil='sobel', th=20, plot=1)
plt.show()

def performance(truth, model):
    truth = truth.astype(bool)
    model = model.astype(bool)
    TP = np.sum(truth * model)
    FP = np.sum(np.invert(truth) * model)
    FN = np.sum(truth * np.invert(model))
    TN = np.sum(np.invert(truth) * np.invert(model))
    confusion_mat = np.array([[TN, FP],[FN, TP]])
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = (TP+TN)/(TP+FN+FP+TN)
    f1 = 2*(precision*recall)/(precision+recall)
    FPR = FP/(TN+FP)
    TPR = TP/(TP+FN)
    return [precision, recall, f1, FPR, TPR, confusion_mat]

pf_prewitt = performance(edge, edge_prewitt)
pf_sobel = performance(edge, edge_sobel)
print(f'Prewitt : (precision, recall, f1)=({pf_prewitt[0]:.4f}, {pf_prewitt[1]:.4f}, {pf_prewitt[2]:.4f})')
print(f'Sobel : (precision, recall, f1)=({pf_sobel[0]:.4f}, {pf_sobel[1]:.4f}, {pf_sobel[2]:.4f})')

def ROC(gray, truth, fil, th_list):
    model = []; FPR = []; TPR = []
    for i in range(len(th_list)):
        model.append(edge_filter(gray, fil=fil, th=th_list[i]))
        pf = performance(truth, model[i])
        FPR.append(pf[3])
        TPR.append(pf[4])
    plt.figure(f'ROC, {fil}, th={th_list}')
    plt.title(f'ROC, {fil}, th={th_list}')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.plot(FPR, TPR)
    plt.scatter(FPR, TPR)
    for i, txt in enumerate(th_list):
        plt.annotate(txt, (FPR[i], TPR[i]))
    #plt.savefig(f'2.2 ROC, {fil}, th={th_list}.png')
    #plt.show()

ROC(gray, edge, 'prewitt', [10, 20, 30, 40, 50])
ROC(gray, edge, 'sobel', [10, 20, 30, 40, 50])
plt.show()

# 2.3) Use machine learning schemes for edge detection.
# 2.3.0) Preprocessing
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
def sub_image(img, K, vector=0):
    sub = np.zeros(img.shape+(K**2,))
    k = K//2
    size = img.size
    img = np.pad(img, k, 'edge')
    for i in range(K):
        for j in range(K):
            sub[:,:,K*i+j] = img[i:img.shape[0]-2*k+i, j:img.shape[1]-2*k+j]
    if vector==1: sub = sub.reshape([size,K**2])
    return sub
# make sub image vector and split data, standardize
X = sub_image(gray, 3, vector=1)
y = binary_image(edge, 200, 1).ravel()
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3)
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)
X = stdsc.transform(X)

# 2.3.a) SVM
from sklearn.svm import SVC
svm = SVC(gamma='auto')
tic = time.time()
svm.fit(X_train, y_train)
y_svm = svm.predict(X_test)
toc = time.time()
pf_svm = performance(y_test, y_svm)
pf_svm.append(toc-tic)
print(f'\nSVM : (precision, recall, f1)=({pf_svm[0]:.4f}, {pf_svm[1]:.4f}, {pf_svm[2]:.4f}), {pf_svm[6]:.3f}sec')

# 2.3.b) SLP
from sklearn.linear_model import Perceptron
ppn = Perceptron()
tic = time.time()
ppn.fit(X_train, y_train)
y_ppn = ppn.predict(X_test)
toc = time.time()
pf_ppn = performance(y_test, y_ppn)
pf_ppn.append(toc-tic)
print(f'SLP : (precision, recall, f1)=({pf_ppn[0]:.4f}, {pf_ppn[1]:.4f}, {pf_ppn[2]:.4f}), {pf_ppn[6]:.3f}sec')

# 2.3.c) MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=300)
tic = time.time()
mlp.fit(X_train, y_train)
y_mlp = mlp.predict(X_test)
toc = time.time()
pf_mlp = performance(y_test, y_mlp)
pf_mlp.append(toc-tic)
print(f'MLP : (precision, recall, f1)=({pf_mlp[0]:.4f}, {pf_mlp[1]:.4f}, {pf_mlp[2]:.4f}), {pf_mlp[6]:.3f}sec')

# 2.3.d) Comparision
edge_svm = svm.predict(X)
edge_svm = edge_svm.reshape(edge.shape)
edge_ppn = ppn.predict(X)
edge_ppn = edge_ppn.reshape(edge.shape)
edge_mlp = mlp.predict(X)
edge_mlp = edge_mlp.reshape(edge.shape)
plt.figure('ML edge detection')
plt.subplot(2,2,1); plt.title('SVM'); plt.axis('off')
plt.imshow(edge_svm, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,2); plt.title('SLP'); plt.axis('off')
plt.imshow(edge_ppn, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,3); plt.title('MLP'); plt.axis('off')
plt.imshow(edge_mlp, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,4); plt.title('Ground truth'); plt.axis('off')
plt.imshow(edge, cmap='gray')
plt.tight_layout()
#plt.savefig('2.3 ML edge detection.png')
plt.show()

# 2.4) Compare the performance of various ML conditions

# 2.4.a) Sub-image(K) size
X_5 = sub_image(gray, 5, vector=1)
y_5 = binary_image(edge, 200, 1).ravel()
X_train_5, X_test_5, y_train_5, y_test_5 = tts(X, y, test_size=0.3)
stdsc = StandardScaler()
X_train_5 = stdsc.fit_transform(X_train_5)
X_test_5 = stdsc.transform(X_test_5)
X_5 = stdsc.transform(X)
svm_5 = SVC(gamma='auto')
tic = time.time()
svm_5.fit(X_train_5, y_train_5)
y_svm_5 = svm_5.predict(X_test_5)
toc = time.time()
pf_svm_5 = performance(y_test_5, y_svm_5)
print(f'\nSVM(k=5) : (precision, recall, f1)=({pf_svm_5[0]:.4f}, {pf_svm_5[1]:.4f}, {pf_svm_5[2]:.4f}), {(toc-tic):.3f}sec')
ppn_5 = Perceptron()
tic = time.time()
ppn_5.fit(X_train_5, y_train_5)
y_ppn_5 = ppn_5.predict(X_test_5)
toc = time.time()
pf_ppn_5 = performance(y_test_5, y_ppn_5)
print(f'SLP(k=5) : (precision, recall, f1)=({pf_ppn_5[0]:.4f}, {pf_ppn_5[1]:.4f}, {pf_ppn_5[2]:.4f}), {(toc-tic):.3f}sec')
mlp_5 = MLPClassifier(hidden_layer_sizes=(10,), max_iter=300)
tic = time.time()
mlp_5.fit(X_train_5, y_train_5)
y_mlp_5 = mlp_5.predict(X_test_5)
toc = time.time()
pf_mlp_5 = performance(y_test_5, y_mlp_5)
print(f'MLP(k=5) : (precision, recall, f1)=({pf_mlp_5[0]:.4f}, {pf_mlp_5[1]:.4f}, {pf_mlp_5[2]:.4f}), {(toc-tic):.3f}sec')

X_7 = sub_image(gray, 7, vector=1)
y_7 = binary_image(edge, 200, 1).ravel()
X_train_7, X_test_7, y_train_7, y_test_7 = tts(X, y, test_size=0.3)
stdsc = StandardScaler()
X_train_7 = stdsc.fit_transform(X_train_7)
X_test_7 = stdsc.transform(X_test_7)
X_7_7 = stdsc.transform(X)
svm_7 = SVC(gamma='auto')
tic = time.time()
svm_7.fit(X_train_7, y_train_7)
y_svm_7 = svm_7.predict(X_test_7)
toc = time.time()
pf_svm_7 = performance(y_test_7, y_svm_7)
print(f'\nSVM(k=7) : (precision, recall, f1)=({pf_svm_7[0]:.4f}, {pf_svm_7[1]:.4f}, {pf_svm_7[2]:.4f}), {(toc-tic):.3f}sec')
ppn_7 = Perceptron()
tic = time.time()
ppn_7.fit(X_train_7, y_train_7)
y_ppn_7 = ppn_7.predict(X_test_7)
toc = time.time()
pf_ppn_7 = performance(y_test_7, y_ppn_7)
print(f'SLP(k=7) : (precision, recall, f1)=({pf_ppn_7[0]:.4f}, {pf_ppn_7[1]:.4f}, {pf_ppn_7[2]:.4f}), {(toc-tic):.3f}sec')
mlp_7 = MLPClassifier(hidden_layer_sizes=(10,), max_iter=300)
tic = time.time()
mlp_7.fit(X_train_7, y_train_7)
y_mlp_7 = mlp_7.predict(X_test_7)
toc = time.time()
pf_mlp_7 = performance(y_test_7, y_mlp_7)
print(f'MLP(k=7) : (precision, recall, f1)=({pf_mlp_7[0]:.4f}, {pf_mlp_7[1]:.4f}, {pf_mlp_7[2]:.4f}), {(toc-tic):.3f}sec')


# 2.4.b) Training data size
X_train_60, X_test_60, y_train_60, y_test_60 = tts(X, y, test_size=0.4)
stdsc = StandardScaler()
X_train_60 = stdsc.fit_transform(X_train_60)
X_test_60 = stdsc.transform(X_test_60)
X_60 = stdsc.transform(X)
svm_60 = SVC(gamma='auto')
tic = time.time()
svm_60.fit(X_train_60, y_train_60)
y_svm_60 = svm_60.predict(X_test_60)
toc = time.time()
pf_svm_60 = performance(y_test_60, y_svm_60)
print(f'\nSVM(train=60%) : (precision, recall, f1)=({pf_svm_60[0]:.4f}, {pf_svm_60[1]:.4f}, {pf_svm_60[2]:.4f}), {(toc-tic):.3f}sec')
ppn_60 = Perceptron()
tic = time.time()
ppn_60.fit(X_train_60, y_train_60)
y_ppn_60 = ppn_60.predict(X_test_60)
toc = time.time()
pf_ppn_60 = performance(y_test_60, y_ppn_60)
print(f'SLP(train=60%) : (precision, recall, f1)=({pf_ppn_60[0]:.4f}, {pf_ppn_60[1]:.4f}, {pf_ppn_60[2]:.4f}), {(toc-tic):.3f}sec')
mlp_60 = MLPClassifier(hidden_layer_sizes=(10,), max_iter=300)
tic = time.time()
mlp_60.fit(X_train_60, y_train_60)
y_mlp_60 = mlp.predict(X_test_60)
toc = time.time()
pf_mlp_60 = performance(y_test_60, y_mlp_60)
print(f'MLP(tran=60%) : (precision, recall, f1)=({pf_mlp_60[0]:.4f}, {pf_mlp_60[1]:.4f}, {pf_mlp_60[2]:.4f}), {(toc-tic):.3f}sec')

X_train_50, X_test_50, y_train_50, y_test_50 = tts(X, y, test_size=0.5)
stdsc = StandardScaler()
X_train_50 = stdsc.fit_transform(X_train_50)
X_test_50 = stdsc.transform(X_test_50)
X_50 = stdsc.transform(X)
svm_50 = SVC(gamma='auto')
tic = time.time()
svm_50.fit(X_train_50, y_train_50)
y_svm_50 = svm_50.predict(X_test_50)
toc = time.time()
pf_svm_50 = performance(y_test_50, y_svm_50)
print(f'\nSVM(train=50%) : (precision, recall, f1)=({pf_svm_50[0]:.4f}, {pf_svm_50[1]:.4f}, {pf_svm_50[2]:.4f}), {(toc-tic):.3f}sec')
ppn_50 = Perceptron()
tic = time.time()
ppn_50.fit(X_train_50, y_train_50)
y_ppn_50 = ppn_50.predict(X_test_50)
toc = time.time()
pf_ppn_50 = performance(y_test_50, y_ppn_50)
print(f'SLP(train=50%) : (precision, recall, f1)=({pf_ppn_50[0]:.4f}, {pf_ppn_50[1]:.4f}, {pf_ppn_50[2]:.4f}), {(toc-tic):.3f}sec')
mlp_50 = MLPClassifier(hidden_layer_sizes=(10,), max_iter=300)
tic = time.time()
mlp_50.fit(X_train_50, y_train_50)
y_mlp_50 = mlp_50.predict(X_test_50)
toc = time.time()
pf_mlp_50 = performance(y_test_50, y_mlp_50)
print(f'MLP(train=50%) : (precision, recall, f1)=({pf_mlp_50[0]:.4f}, {pf_mlp_50[1]:.4f}, {pf_mlp_50[2]:.4f}), {(toc-tic):.3f}sec')



# 2.4.c) SVM kernel
print(f'\nSVM(rbf) : (precision, recall, f1)=({pf_svm[0]:.4f}, {pf_svm[1]:.4f}, {pf_svm[2]:.4f}), {pf_svm[6]:.3f}sec')
svm_sig = SVC(kernel='sigmoid', gamma='auto')
tic = time.time()
svm_sig.fit(X_train, y_train)
y_svm_sig  = svm_sig.predict(X_test)
toc = time.time()
pf_svm_sig = performance(y_test, y_svm_sig)
print(f'SVM(sigmoid) : (precision, recall, f1)=({pf_svm_sig[0]:.4f}, {pf_svm_sig[1]:.4f}, {pf_svm_sig[2]:.4f}), {(toc-tic):.3f}sec')
edge_svm_sig = svm_sig.predict(X)
edge_svm_sig = edge_svm_sig.reshape(edge.shape)
plt.figure('SVM kernel')
plt.subplot(1,3,1); plt.title('SVM(rbf)'); plt.axis('off')
plt.imshow(edge_svm, cmap='gray', vmin=0, vmax=1)
plt.subplot(1,3,2); plt.title('SVM(sigmoid)'); plt.axis('off')
plt.imshow(edge_svm_sig, cmap='gray', vmin=0, vmax=1)
plt.subplot(1,3,3); plt.title('Ground truth'); plt.axis('off')
plt.imshow(edge, cmap='gray')
plt.tight_layout()
#plt.savefig('2.4.c SVM kernel.png')
plt.show()

# 2.4.b) MLP layer
print(f'\nMLP(10) : (precision, recall, f1)=({pf_mlp[0]:.4f}, {pf_mlp[1]:.4f}, {pf_mlp[2]:.4f}), {pf_mlp[6]:.3f}sec')
mlp_10_2 = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=300)
tic = time.time()
mlp_10_2.fit(X_train, y_train)
y_mlp_10_2 = mlp_10_2.predict(X_test)
toc = time.time()
pf_mlp_10_2 = performance(y_test, y_mlp_10_2)
print(f'MLP(10,10) : (precision, recall, f1)=({pf_mlp_10_2[0]:.4f}, {pf_mlp_10_2[1]:.4f}, {pf_mlp_10_2[2]:.4f}), {(toc-tic):.3f}sec')
mlp_10_3 = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=300)
tic = time.time()
mlp_10_3.fit(X_train, y_train)
y_mlp_10_3 = mlp_10_3.predict(X_test)
toc = time.time()
pf_mlp_10_3 = performance(y_test, y_mlp_10_3)
print(f'MLP(10,10,10) : (precision, recall, f1)=({pf_mlp_10_3[0]:.4f}, {pf_mlp_10_3[1]:.4f}, {pf_mlp_10_3[2]:.4f}), {(toc-tic):.3f}sec')
edge_mlp_10_2 = mlp_10_2.predict(X)
edge_mlp_10_2 = edge_mlp_10_2.reshape(edge.shape)
edge_mlp_10_3 = mlp_10_2.predict(X)
edge_mlp_10_3 = edge_mlp_10_3.reshape(edge.shape)
plt.figure('MLP layer')
plt.subplot(2,2,1); plt.title('MLP(10)'); plt.axis('off')
plt.imshow(edge_mlp, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,2); plt.title('MLP(10,10)'); plt.axis('off')
plt.imshow(edge_mlp_10_2, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,3); plt.title('MLP(10,10,10)'); plt.axis('off')
plt.imshow(edge_mlp_10_3, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,4); plt.title('Ground truth'); plt.axis('off')
plt.imshow(edge, cmap='gray')
plt.tight_layout()
#plt.savefig('2.4.d MLP layer.png')
plt.show()

# 2.4.e) PCA
from sklearn.decomposition import PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train)
plt.figure('PCA Explained Variance Ratio')
plt.title('PCA Explained Variance Ratio')
plt.bar(range(0, 9), pca.explained_variance_ratio_)
plt.step(range(0, 9), np.cumsum(pca.explained_variance_ratio_))
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
#plt.savefig('2.4.e) PCA EVR.png')
plt.show()
pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
X_pca = pca.transform(X)
svm_pca = SVC(gamma='auto')
tic = time.time()
svm_pca.fit(X_train_pca, y_train)
y_svm_pca = svm_pca.predict(X_test_pca)
toc = time.time()
pf_svm_pca = performance(y_test, y_svm_pca)
print(f'\nSVM : (precision, recall, f1)=({pf_svm[0]:.4f}, {pf_svm[1]:.4f}, {pf_svm[2]:.4f}), {pf_svm[6]:.3f}sec')
print(f'SVM with PCA : (precision, recall, f1)=({pf_svm_pca[0]:.4f}, {pf_svm_pca[1]:.4f}, {pf_svm_pca[2]:.4f}), {(toc-tic):.3f}sec')
ppn_pca = Perceptron()
tic = time.time()
ppn_pca.fit(X_train_pca, y_train)
y_ppn_pca = ppn_pca.predict(X_test_pca)
toc = time.time()
pf_ppn_pca = performance(y_test, y_ppn_pca)
print(f'\nSLP : (precision, recall, f1)=({pf_ppn[0]:.4f}, {pf_ppn[1]:.4f}, {pf_ppn[2]:.4f}), {pf_ppn[6]:.3f}sec')
print(f'SLP with PCA : (precision, recall, f1)=({pf_ppn_pca[0]:.4f}, {pf_ppn_pca[1]:.4f}, {pf_ppn_pca[2]:.4f}), {(toc-tic):.3f}sec')
mlp_pca = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=300)
tic = time.time()
mlp_pca.fit(X_train_pca, y_train)
y_mlp_pca = mlp_pca.predict(X_test_pca)
toc = time.time()
pf_mlp_pca = performance(y_test, y_mlp_pca)
print(f'\nMLP : (precision, recall, f1)=({pf_mlp[0]:.4f}, {pf_mlp[1]:.4f}, {pf_mlp[2]:.4f}), {pf_mlp[6]:.3f}sec')
print(f'MLP with PCA : (precision, recall, f1)=({pf_mlp_pca[0]:.4f}, {pf_mlp_pca[1]:.4f}, {pf_mlp_pca[2]:.4f}), {(toc-tic):.3f}sec')
edge_svm_pca = svm_pca.predict(X_pca)
edge_svm_pca = edge_svm_pca.reshape(edge.shape)
edge_ppn_pca = ppn_pca.predict(X_pca)
edge_ppn_pca = edge_ppn_pca.reshape(edge.shape)
edge_mlp_pca = mlp_pca.predict(X_pca)
edge_mlp_pca = edge_mlp_pca.reshape(edge.shape)
plt.figure('PCA')
plt.subplot(2,2,1); plt.title('SVM with PCA'); plt.axis('off')
plt.imshow(edge_svm_pca, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,2); plt.title('SLP with PCA'); plt.axis('off')
plt.imshow(edge_ppn_pca, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,3); plt.title('MLP with PCA'); plt.axis('off')
plt.imshow(edge_mlp_pca, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,4); plt.title('Ground truth'); plt.axis('off')
plt.imshow(edge, cmap='gray')
plt.tight_layout()
#plt.savefig('2.4.e PCA.png')
plt.show()

# 2.4.f) Apply on other image
img_ = cv2.imread('pepper.bmp')
gray_ = B2G(img_)
X_ = sub_image(gray_, 3, vector=1)
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_ = stdsc.transform(X_)
edge_ = cv2.Canny(gray_, 100, 150)
tic = time.time()
edge_svm_ = svm.predict(X_)
toc = time.time()
edge_svm_ = edge_svm_.reshape(gray_.shape)
print(f'\nSVM predict time = {(toc-tic):.3f}')
tic = time.time()
edge_ppn_ = ppn.predict(X_)
toc = time.time()
edge_ppn_ = edge_ppn_.reshape(gray_.shape)
print(f'\nSLP predict time = {(toc-tic):.3f}')
tic = time.time()
edge_mlp_ = mlp.predict(X_)
toc = time.time()
edge_mlp_ = edge_mlp_.reshape(gray_.shape)
print(f'\nMLP predict time = {(toc-tic):.3f}')
plt.figure('Apply')
plt.subplot(2,2,1); plt.title('SVM'); plt.axis('off')
plt.imshow(edge_svm_, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,2); plt.title('SLP'); plt.axis('off')
plt.imshow(edge_ppn_, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,3); plt.title('MLP'); plt.axis('off')
plt.imshow(edge_mlp_, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,4); plt.title('Ground truth'); plt.axis('off')
plt.imshow(edge_, cmap='gray')
plt.tight_layout()
#plt.savefig('2.4.f) pepper ML.png')
X_pca_ = pca.transform(X_)
tic = time.time()
edge_svm_pca_ = svm_pca.predict(X_pca_)
toc = time.time()
edge_svm_pca_ = edge_svm_pca_.reshape(gray_.shape)
print(f'\nSVM_PCA predict time = {(toc-tic):.3f}')
tic = time.time()
edge_ppn_pca_ = ppn_pca.predict(X_pca_)
toc = time.time()
edge_ppn_pca_ = edge_ppn_pca_.reshape(gray_.shape)
print(f'\nSLP_PCA predict time = {(toc-tic):.3f}')
tic = time.time()
edge_mlp_pca_ = mlp_pca.predict(X_pca_)
toc = time.time()
edge_mlp_pca_ = edge_mlp_pca_.reshape(gray_.shape)
print(f'\nMLP_PCA predict time = {(toc-tic):.3f}')
plt.figure('Apply_PCA')
plt.subplot(2,2,1); plt.title('SVM with PCA'); plt.axis('off')
plt.imshow(edge_svm_pca_, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,2); plt.title('SLP with PCA'); plt.axis('off')
plt.imshow(edge_ppn_pca_, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,3); plt.title('MLP with PCA'); plt.axis('off')
plt.imshow(edge_mlp_pca_, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,2,4); plt.title('Ground truth'); plt.axis('off')
plt.imshow(edge_, cmap='gray')
plt.tight_layout()
#plt.savefig('2.4.f) pepper ML_PCA.png')
plt.show()
