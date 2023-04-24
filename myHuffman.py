import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functools import total_ordering
import bisect
import numpy as np

from numpy import ndarray


def levelFreqDict(objs : ndarray):
    #levelFreqs = dict.fromkeys([i for i in range(min(imBytes), max(imBytes)+1)], 0)
    levelFreqs = dict.fromkeys(list(set(objs)), 0)
    for i in objs:
        levelFreqs[i] += 1
    return levelFreqs

def treeFromDict(levelFreqs: dict):
    # Huffman-encode Gray-Level Histogram, Source: Module 7 pp 25
    # Create leaf node for all pixel-values
    nodes = [node(levelFreqs[i], i, None, None ) for i in levelFreqs]

    # Debug helper. sumprob should equal size of original image
    sumprob = 0
    for i in nodes:
        sumprob = sumprob + i.prob

    # Connect nodes into a Huffman Tree
    nodes.sort()
    while len(nodes) > 1:
        lefty = nodes.pop(0)
        righty = nodes.pop(0)
        newNode = node(lefty.prob + righty.prob, None, lefty, righty)
        bisect.insort(nodes, newNode) # Ordered Insert
    
    if len(nodes) == 1:
        return nodes[0], sumprob
    else:
        raise Exception(f"# of nodes is != 1. len(nodes) == {len(nodes)}")

# get level->code conversion dictionary from a root node
def conversionDict(root, img):
    codes = {}
    root.returnCodes([], codes)

    return codes

def compressAndPackage(img, codes, root, verbose=False):
    # convert bytes into array of codes
    #   this is an array of arrays, sub array elements are int-1 or int-0
    codeArray = [codes[b] for b in img]

    # Keep as arrays until the number of zeros lost in the integer conversion is found
    # If a run of N zeros makes up the most significant bits, they will be lost in integer conversion
    #   So a zeropad variable stores the number of leading zeros, if applicable
    zeroPad = 0
    breakOut = False
    for code in reversed(codeArray):
        for bits in reversed(code):
            if bits == 0:
                zeroPad += 1
            else:
                breakOut = True
                break
        if breakOut:
            break

    # Calculate BPP (bytes per pixel) and print compression ratio
    bitsCount = 0
    for code in codeArray:
        bitsCount += len(code)
    bytesCount = bitsCount / 8

    # Create actual bytes object with zero-padding, data, and packaged tree
    idx = 0
    number = 0
    for code in codeArray:
        for bit in code:
            number += bit << idx
            idx += 1
    if verbose:
        print(f"Verify the bit count == compressed bin representation length + zero pad count:\n"
              f"    {bitsCount} =?= {len(bin(number)) - 2} + {zeroPad}")

    pack = {"Pad" : zeroPad, "Bits" : number, "Tree" : root}
    return pack

def hftCompress(objs, verbose=False, bytesPerObj=None):
    objFreqs = levelFreqDict(objs)
    hftRoot, junk = treeFromDict(objFreqs)
    codeDict = conversionDict(hftRoot, objs)
    pack = compressAndPackage(objs, codeDict, hftRoot, verbose)
    if verbose:
        printStats(objs, pack["Bits"], bytesPerObj)
    return pack

def hftDecompress(binary, zeroPad, tree):
    shiftedBin = binary >> 0
    nextBit = None
    nextNode = tree
    levels = []
    while shiftedBin > 0:
        nextBit = shiftedBin & 1
        nextNode = nextNode.left if nextBit == 0 else nextNode.right
        if nextNode.value != None:
            levels.append(nextNode.value)
            nextNode = tree
        shiftedBin = shiftedBin >> 1
    
    zp = zeroPad
    while zp > 0:
        nextNode = nextNode.left
        if nextNode.value != None:
            levels.append(nextNode.value)
            nextNode = tree
        zp -= 1

    return levels

def printStats(objs, objBinary, bytesPerObj=None):
    if bytesPerObj != None:
        compressedBytesPerObj = (len(bin(objBinary)) - 2) / 8 / len(objs)
        print(f"Original Bytes per Pixel: {bytesPerObj}\n"
            f"Compressed Bytes per Pixel: {compressedBytesPerObj}\n"
            f"Compression Ratio: {bytesPerObj / compressedBytesPerObj}")
    else:
        print(f"Encoded Object Count: {bytesPerObj}\n"
            f"Compressed Bytes per Pixel: {compressedBytesPerObj}\n")
        
#TODO: implement pickle file creator
    


@total_ordering
class node:
    def __init__(self, prob, value=None, left=None, right=None):
        self.prob = prob
        self.value = value
        self.left = left
        self.right = right
        self.ancestry = []
    
    def print(self):
        print(
            "Val:", self.value,
            ", Freq:", self.prob,
            ", Left:", self.left.value if self.left else '__', self.left.prob if self.left else '__',
            ", Right:", self.right.value if self.right else '__', self.right.prob if self.right else '__'
            )
        if self.left: self.left.print()
        if self.right: self.right.print()

    def returnCodes(self, bits, codesPtr):
        self.ancestry = bits
        if self.value != None:
            codesPtr[self.value] = bits
        else:
            if self.left:
                leftBits = self.ancestry.copy()
                leftBits.append(0)
                self.left.returnCodes(leftBits,codesPtr)
            if self.right:
                rightBits = self.ancestry.copy()
                rightBits.append(1)
                self.right.returnCodes(rightBits, codesPtr)


    def __repr__(self):
        return f"(freq: {self.prob}, val: {self.value}, path: {self.ancestry})"

    def __eq__(self, otherNode):
        return self.prob == otherNode.prob
    
    def __lt__(self, otherNode):
        return  self.prob < otherNode.prob

def test():
    # Huffman Encode
    with open('camera.bin', 'rb') as file:
        imBytes = file.read()
    imArr = np.array([i for i in imBytes], dtype=np.uint8)
    img = imArr.reshape(256, 256)

    plt.figure()
    imgplot = plt.imshow(img, cmap='gray')
    plt.title("Original Cameraman Image")


    #Create and display Pixel Dictionary
    levelFreqs = levelFreqDict(img.tobytes())
    plt.figure()
    imHist = plt.bar(levelFreqs.keys(), levelFreqs.values())
    plt.title("Image Gray-Level Histogram")

    # hftRoot, junk = hfTree.treeFromDict(levelFreqs)
    # codeDict = hfTree.conversionDict(hftRoot, img)
    package = hftCompress(img.tobytes(), verbose=True, bytesPerObj=1)
    reconstruct = hftDecompress(package["Bits"], package["Pad"], package["Tree"])

    # print(f"Verifying Lossless: {all(reconstruct == imArr)}")
    img_reconstruct = np.array([i for i in reconstruct], dtype=np.uint8).reshape(256,256)
    plt.figure()
    plt.title("Reconstructed Cameraman Image")
    imgplot = plt.imshow(img, cmap='gray')

    # Calculate MSE (hopefully Zero)
    mse = 0
    for i, j in np.ndindex(img_reconstruct.shape):
        mse += (img_reconstruct[i, j] - img[i, j]) ** 2 / 65536
    print(f"MSE: {mse}")


    plt.show()

if __name__ == "__main__":
    test()