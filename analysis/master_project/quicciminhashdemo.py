def printDescriptor(descriptor):
    for i in range(len(descriptor)-1, -1, -1):
        print(descriptor[i])

def generateSignature(descriptor, perms):
    signature = []
    for p in perms:
        for index, value in enumerate(p):
            if (descriptor[math.floor(value/len(descriptor))][value % len(descriptor)]) == "X":
                signature.append(index)
                break
    return signature

def descriptorToString(descriptor):
    string = ""
    for row in range(4):
        for col in range(4):
            string += "1" if descriptor[row][col] == "X" else "0"
    return string

permutations = [
    [8, 2, 13, 0, 12, 10, 3, 5, 9, 14, 7, 4, 6, 11, 1, 15],
    [10, 13, 4, 7, 3, 0, 14, 6, 9, 12, 11, 1, 5, 8, 15, 2],
    [3, 14, 7, 4, 9, 1, 5, 6, 11, 15, 0, 13, 10, 8, 2, 12]
]

"""
for i in range(16):

    for p in permutations:
        print(p[i], end=" ")
    print()
"""

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

print(descriptorToString(descriptor1))
print(descriptorToString(descriptor2))

#print(generateSignature(descriptor1, permutations))
#print(generateSignature(descriptor2, permutations))