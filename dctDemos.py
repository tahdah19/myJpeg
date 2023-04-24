import numpy as np
from scipy.fft import dctn, idctn
import matplotlib.pyplot as plt
import gzip

from myjpeg import quantizedDct, decodeQDct, normArr
import myHuffman

np.set_printoptions(suppress=True)

# *************************************
# DCT QUANTIZATION DEMO
# *************************************
normArr = np.array([[16, 11, 10, 16,  24,  40,  51,  61],
                    [12, 12, 14, 19,  26,  58,  60,  55],
                    [14, 13, 16, 24,  40,  57,  69,  56],
                    [14, 17, 22, 29,  51,  87,  80,  62],
                    [18, 22, 37, 56,  68, 109, 103,  77],
                    [24, 35, 55, 64,  81, 104, 113,  92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103,  99]])

with open('camera.bin', 'rb') as file:
    imBytes = file.read()
imArr = np.array([i for i in imBytes], dtype=np.int16)
img = imArr.reshape(256, 256)

subImg = img[56:64,120:128]
plt.figure()
plt.imshow(subImg, cmap='gray')
plt.title("Cameraman.bin Subset")
plt.show()

subDctMag = np.abs(dctn(subImg, norm='ortho', orthogonalize=True))
plt.imshow(subDctMag, cmap='gray')
plt.title("8x8 DCT Block (absolute values)")
plt.show()

subDctMagQuant = np.round(subDctMag / normArr)
plt.imshow(subDctMag, cmap='gray')
plt.title("8x8 DCT Quantized (absolute values)")
plt.show()

# *************************************
# DCT DEMO BLOCKS
# *************************************
# dctDemo1 = np.array([[255 for i in range(0, 8*8)]], dtype=np.int16).reshape(8,8)
# dctDemo2 = np.array([[255, 0, 255, 0, 255, 0, 255, 0] for i in range(0,8)], dtype=np.int16)
# dctDemo3 = np.array([[255, 255, 255, 255, 255, 255, 255, 255],
#                      [255, 255, 255, 255, 255, 255, 255, 255],
#                      [255, 255, 170, 100, 100, 170, 255, 255],
#                      [255, 255, 100, 0  , 0  , 100, 255, 255],
#                      [255, 255, 100, 0  , 0  , 100, 255, 255],
#                      [255, 255, 170, 100, 100, 170, 255, 255],
#                      [255, 255, 255, 255, 255, 255, 255, 255],
#                      [255, 255, 255, 255, 255, 255, 255, 255],], dtype=np.int16)

# for i in [dctDemo1, dctDemo2, dctDemo3]:
#     print(np.round(dctn(i, norm='ortho', orthogonalize=True)))
#     plt.imshow(i, cmap='gray')
#     plt.show()

# plt.imshow(dctn(dctDemo2, norm='ortho', orthogonalize=True), cmap='gray')
# plt.show()

# *************************************
# DCT DC CONTINUITY DEMO
# *************************************
# with open('camera.bin', 'rb') as file:
#     imBytes = file.read()
# imArr = np.array([i for i in imBytes], dtype=np.int16)
# # imArr = exBlock
# img = imArr.reshape(256, 256)
# plt.figure()
# plt.imshow(img, cmap='gray')
# plt.title("Original Cameraman.bin")
# plt.show()

# dcDemoBlock1 = img[136:145,46:55]
# dcDemoBlock2 = img[136:145,56:65]
# print(dctn(dcDemoBlock1, norm='ortho', orthogonalize=True)[0,0])
# print(dctn(dcDemoBlock2, norm='ortho', orthogonalize=True)[0,0])

# cameramanQuant = quantizedDct(img, 256, 256)
# cameramanDCs = [cameramanQuant[i,j][0,0] for i, j in np.ndindex(cameramanQuant.shape)]
# cameramanHist = {i:0 for i in range(min(cameramanDCs), max(cameramanDCs)+1)}
# for i in cameramanDCs:
#     cameramanHist[i] += 1
# plt.bar(cameramanHist.keys(), cameramanHist.values())
# plt.title("Cameraman Quantized DC Histogram")
# plt.show()

# dcHftPack = myHuffman.hftCompress(cameramanDCs)
# print("Bit Count = ", dcHftPack["Pad"] + len(bin(dcHftPack["Bits"])) - 2)

# f1 = gzip.GzipFile("dcs.npy.gz", "w")
# np.save(f1, np.array(cameramanDCs))

# cameramanDCsDiff = [cameramanDCs[i] - cameramanDCs[i-1] for i in range(1, len(cameramanDCs))]
# cameramanDiffHist = {i:0 for i in range(min(cameramanDCsDiff[1:]), max(cameramanDCsDiff[1:])+1)}
# for i in cameramanDCsDiff:
#     cameramanDiffHist[i] += 1
# plt.bar(cameramanDiffHist.keys(), cameramanDiffHist.values())
# plt.title("Cameraman Quantized DC Differences Histogram")
# plt.show()

# dcDiffHftPack = myHuffman.hftCompress(cameramanDCsDiff)
# print("Bit Count = ", dcDiffHftPack["Pad"] + len(bin(dcDiffHftPack["Bits"])) - 2)

# f2 = gzip.GzipFile("dcdiffs.npy.gz", "w")
# np.save(f2, np.array(cameramanDCsDiff))

# *************************************
# gzipping
# *************************************
# with open('camera.bin', 'rb') as file:
#     imBytes = file.read()
# imArr = np.array([i for i in imBytes], dtype=np.int16)
# img = imArr.reshape(256, 256)

# np.save("savedArray1.npy", img)

# f = gzip.GzipFile("savedArray2.npy.gz", "w")
# np.save(f, img)

# *************************************
# DCT example generation
# *************************************
# dctDemo1 = np.array([[255 for i in range(0, 8*8)]], dtype=np.int16).reshape(8,8)
# dctDemo2 = np.array([[255, 0, 255, 0, 255, 0, 255, 0] for i in range(0,8)], dtype=np.int16)
# dctDemo3 = np.array([[255, 255, 255, 255, 255, 255, 255, 255],
#                     [255, 255, 255, 255, 255, 255, 255, 255],
#                     [255, 255, 170, 100, 100, 170, 255, 255],
#                     [255, 255, 100, 0  , 0  , 100, 255, 255],
#                     [255, 255, 100, 0  , 0  , 100, 255, 255],
#                     [255, 255, 170, 100, 100, 170, 255, 255],
#                     [255, 255, 255, 255, 255, 255, 255, 255],
#                     [255, 255, 255, 255, 255, 255, 255, 255],], dtype=np.int16)

# for i in [dctDemo1, dctDemo2, dctDemo3]:
#     print(np.round(dctn(i, norm='ortho', orthogonalize=True)))
#     plt.imshow(i, cmap='gray')
#     plt.show()

# plt.imshow(dctn(dctDemo2, norm='ortho', orthogonalize=True), cmap='gray')
# plt.show()

# *************************************
# Image Freqs example generation
# *************************************
# lowFreq1d = (np.cos(0.0125*np.array([i for i in range(0,256)])) + 1) * 127
# plt.subplot(1, 2, 1)
# plt.plot(lowFreq1d)
# plt.title("Low 1D Frequency")

# lowFreq2d = np.array([lowFreq1d for i in range(0,256)])
# plt.subplot(1, 2, 2)
# plt.imshow(lowFreq2d, cmap='gray')
# plt.title("Low Horizontal Frequency")

# plt.show()

# hiFreq1d = (np.cos(0.5*np.array([i for i in range(0,256)])) + 1) * 127
# plt.subplot(1, 2, 1)
# plt.plot(hiFreq1d)
# plt.title("High 1D Frequency")

# hiFreq2d = np.array([hiFreq1d for i in range(0,256)])
# hiFreq2d = hiFreq2d.transpose()
# plt.subplot(1, 2, 2)
# plt.imshow(hiFreq2d, cmap='gray')
# plt.title("High Vertical Frequency")

# plt.show()