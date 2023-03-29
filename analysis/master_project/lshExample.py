import math
import random

def getRandomList(n):
    l = [x for x in range(0, n+1)]
    random.shuffle(l)
    return l

def printDescriptor(descriptor):
    for i in range(len(descriptor)-1, -1, -1):
        print(descriptor[i])

def printPermutations(permutations):
    for p in permutations:
        print(p)

def computeJaccardSimilarity(sig1, sig2):
    assert len(sig1) == len(sig2)
    count = 0
    for i in range(0, len(sig1)):
        if sig1[i] == sig2[i]:
            count += 1
    return count/len(sig1)

def generateSignature(descriptor, perms):
    signature = []
    for p in perms:
        for index, value in enumerate(p):
            if (descriptor[math.floor(value/len(descriptor))][value % len(descriptor)]) == "X":
                signature.append(index)
                break
    return signature


# DESCRIPTORS
descriptor1 = [
    '   X',
    '  XX',
    ' XX ',
    'X   '
]

descriptor2 = [
    '  XX',
    '  X ',
    ' X  ',
    'X   '
]

descriptor1.reverse()
descriptor2.reverse()

print("Descriptor 1")
printDescriptor(descriptor1)
print()

print("Descriptor 2")
printDescriptor(descriptor2)
print()

similarities = []

for i in range(0,1000):
    # PERMUTATIONS
    permutations = []
    for _ in range(0, 2):
        permutations.append(getRandomList(15))


    # SIGNATURE GENERATION
    signature1 = generateSignature(descriptor1, permutations)
    signature2 = generateSignature(descriptor2, permutations)

    # print(signature1)
    # print(signature2)

    # SIGNATURE SIMILARITY
    similarities.append(computeJaccardSimilarity(signature1, signature2))

avg_sim = sum(similarities)/len(similarities)
true_sim = 4/7

print(avg_sim)
print(abs(avg_sim-true_sim))