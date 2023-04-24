import numpy as np
from scipy.fft import dctn, idctn
import matplotlib.pyplot as plt

import zlib

normArr = np.array([[16, 11, 10, 16,  24,  40,  51,  61],
                    [12, 12, 14, 19,  26,  58,  60,  55],
                    [14, 13, 16, 24,  40,  57,  69,  56],
                    [14, 17, 22, 29,  51,  87,  80,  62],
                    [18, 22, 37, 56,  68, 109, 103,  77],
                    [24, 35, 55, 64,  81, 104, 113,  92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103,  99]])

np.set_printoptions(suppress=True)

def quantizedDct(img, img_w, img_l):

    # img to blocks
    imgBlocks = np.empty(shape=(img_w*img_l//64, 8, 8))
    for i, j in np.ndindex((img_w//8,img_l//8)):
        block = img[i*8:i*8+8,j*8:j*8+8]
        imgBlocks[i*32 + j] = block
    # print(block.dtype)

    # encode blocks
    imgFreqBlocks = dctn(imgBlocks-128, norm='ortho', orthogonalize=True, axes=(1, 2))

    # imgFreqBlocks = np.empty(shape=(img_w//8, img_l//8), dtype=object)
    # for i, j in np.ndindex(imgBlocks.shape):
    #     imgFreqBlocks[i, j] = np.round(dctn(imgBlocks[i, j] - 128, norm="ortho", orthogonalize=True), 1)
    # print(imgFreqBlocks.dtype)

    # quantized, normalized blocks
    imgFreqQuantBlocks = np.round(imgFreqBlocks / normArr)
    # for i in imgFreqQuantBlocks:
    #     print(i)
    # imgFreqQuantBlocks = np.empty(shape=(img_w//8, img_l//8), dtype=object)
    # for i, j in np.ndindex(imgFreqBlocks.shape):
    #     imgFreqQuantBlocks[i, j] = np.array(np.round(imgFreqBlocks[i, j] / normArr), dtype=np.int8)
    # # print(imgFreqQuantBlocks[0,0])
    return imgFreqQuantBlocks.astype(np.int16)

'''
zigzag algorithm
Description from: http://www.rosettacode.org/wiki/Zig_Zag
Published at: https://paddy3118.blogspot.com/2008/08/zig-zag.html
Code Author: Donald 'Paddy' McCarthy
'''
def zigzag(n):
    indexorder = sorted(((x,y) for x in range(n) for y in range(n)),
    key = lambda p: (p[0]+p[1], -p[1] if (p[0]+p[1]) % 2 else p[1]) )
    return dict((index,n) for n,index in enumerate(indexorder))

def acZigZagArr(quantFreqArray):
    zzDict = zigzag(8)
    acArray = np.empty(shape=63, dtype=np.int16)
    for i, j in np.ndindex(quantFreqArray.shape):
        if i == 0 and j == 0: continue
        acArray[zzDict[(i,j)]-1] = quantFreqArray[i, j]
    return acArray.astype(int)

def acDcSeparation(imgFreqQuantBlocks, badAcOrder=False):
    blockArray = imgFreqQuantBlocks
    dcArray = np.empty(shape=(imgFreqQuantBlocks.shape[0]), dtype=np.int16)
    acsArray = np.empty(shape=(imgFreqQuantBlocks.shape[0], 63), dtype=np.int16)

    for i in range(0, blockArray.shape[0]):
        dcArray[i] = blockArray[i][0,0]
        acsArray[i] = blockArray[i].reshape(64,)[1:] if badAcOrder else acZigZagArr(blockArray[i])

    return dcArray, acsArray

def acDcReconstruction(dcArray, acsArray, badAcOrder=False):
    blocks = np.empty(shape=(1024,8,8), dtype=np.int16)
    blocks[:,0,0] = dcArray
    if badAcOrder:
        for b, ac in zip(blocks, acsArray):
            for i, j in np.ndindex(b.shape):
                if i == 0 and j == 0: continue
                b[i,j] = ac[i*8+j-1]
    else:
        zzDict = zigzag(8)
        for b, ac in zip(blocks, acsArray):
            for i, j in np.ndindex(b.shape):
                if i == 0 and j == 0: continue
                b[i,j] = ac[zzDict[(i,j)]-1]
    return blocks


def decodeQDct(imgFreqQuantBlocks, img_w, img_l):

    # denormalize dct blocks
    imgDenormBlocks = imgFreqQuantBlocks * normArr
    # imgDenormBlocks = np.empty(shape=(img_w//8, img_l//8), dtype=object)
    # for i, j in np.ndindex(imgFreqQuantBlocks.shape):
    #     imgDenormBlocks[i, j] = imgFreqQuantBlocks[i, j] * normArr
    # print(imgDenormBlocks[0,0])

    # decode blocks
    imgDecodeBlocks = idctn(imgDenormBlocks, norm="ortho", orthogonalize=True, axes=(1,2)) + 128
    # imgDecodeBlocks = np.empty(shape=(img_w//8, img_l//8), dtype=object)
    # for i, j in np.ndindex(imgDenormBlocks.shape):
    #     imgDecodeBlocks[i, j] = idctn(imgDenormBlocks[i, j], norm="ortho", orthogonalize=True) + 128
    # print(imgDecodeBlocks[0,0])

    # decoded blocks back to image
    imgRcnstd = np.empty(shape=(img_w, img_l))
    for i, j in np.ndindex(img_w//8, img_l//8):
        imgRcnstd[i*8:i*8+8,j*8:j*8+8] = imgDecodeBlocks[i*32 + j] #if np.max(imgDecodeBlocks[i,j]) < 256 else imgDecodeBlocks[i,j] * 
    print(np.min(imgRcnstd), np.max(imgRcnstd))

    return imgRcnstd

if __name__ == "__main__":
  
    with open('peppers.bin', 'rb') as file:
        imBytes = file.read()
    imArr = np.array([i for i in imBytes], dtype=np.uint8)
    img = imArr.reshape(256, 256)

    # Huffman Only Demo
    huffmanOnlyBytes = zlib.compressobj(level=9, strategy=zlib.Z_HUFFMAN_ONLY).compress(imBytes)
    print(f"HUFFMAN ONLY:\noriginal size={len(imArr)}, zlib Huffman Compressed={len(huffmanOnlyBytes)}\n")

    # Quantize Image DCTs
    imgQuant = quantizedDct(img, 256, 256)
    quantObj = zlib.compressobj(level=9, strategy=zlib.Z_HUFFMAN_ONLY)
    quantHuffmaned = quantObj.compress(imgQuant.tobytes())
    quantHuffmaned += quantObj.flush()
    print(f"HUFFMAN ON DCT BLOCKS:\noriginal size={len(imArr)}, arrys sizes={len(quantHuffmaned)}\n")

    # Separate ACs and DCs
    dc, acs = acDcSeparation(imgQuant, badAcOrder=False)
    dcObj = zlib.compressobj(level=9, strategy=zlib.Z_HUFFMAN_ONLY)
    dcHuffmaned = dcObj.compress(dc.tobytes())
    dcHuffmaned += dcObj.flush()
    acObj = zlib.compressobj(level=9, strategy=zlib.Z_HUFFMAN_ONLY)
    acsHuffmaned = acObj.compress(acs.tobytes())
    acsHuffmaned += acObj.flush()
    print(f"HUFFMAN ON AC/DC ARRAYS:\noriginal size={len(imArr)}, arrays sizes={len(dcHuffmaned)} + {len(acsHuffmaned)} = {len(acsHuffmaned)+len(dcHuffmaned)}\n")
    
    # DC RLC
    dcDiff = np.empty(shape=dc.shape, dtype=np.int16)
    dcDiff[0] = dc[0]
    for i in range(1, dcDiff.size):
        dcDiff[i] = dc[i] - dc[i-1]
    dcDiffObj = zlib.compressobj(level=9, strategy=zlib.Z_HUFFMAN_ONLY)
    dcDiffHuffmaned = dcDiffObj.compress(dcDiff.tobytes()) + dcDiffObj.flush()
    acZigObj = zlib.compressobj(level=9, strategy=zlib.Z_RLE)
    acZigHuffmaned = acZigObj.compress(acs.tobytes()) + acZigObj.flush()
    print(f"''FULL'' IMPLEMENTATION:\noriginal size={len(imArr)}, arrays sizes=arrays sizes={len(dcDiffHuffmaned)} + {len(acZigHuffmaned)} = {len(acZigHuffmaned)+len(dcDiffHuffmaned)}\n")

    # Reconstruct Image
    imgRcnstd = decodeQDct(acDcReconstruction(dc, acs, badAcOrder=False), 256, 256)
    imgRcnstd = np.clip(imgRcnstd,0,255)
    print("average error:", np.sum(np.abs(img-imgRcnstd)) / (256*256))

    
    # Display reconstructed image
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(imgRcnstd, cmap='gray')
    plt.title("Reconstructed peppers.bin")
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.title("Original peppers.bin")
    plt.show()

    
