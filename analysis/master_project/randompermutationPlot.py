import matplotlib.pyplot as plt
import random

# Script for verifying the random distribution of permutation values
# Should be able to use this to visualize the distribution of the actual permutations 

numPermutations = 10
descriptorResolution = 32
dic = {}

sumIndexes = [0]*1024




for i in range(numPermutations):
    indexes = list(range(0,descriptorResolution**2))
    random.shuffle(indexes)
    dic[i] = indexes.copy()


for j in range(descriptorResolution**2):
    for i in range(numPermutations):
        sumIndexes[j] += dic[i][j]

print(sumIndexes)


cmap = plt.cm.get_cmap('hsv',max(sumIndexes)*1.1)

for index, value in enumerate(sumIndexes):
    plt.plot(index % descriptorResolution, index // descriptorResolution, marker="s", color=cmap(value))
plt.show()

# Implement a plot which shows the sum of the permutations sequences, as a heatmap