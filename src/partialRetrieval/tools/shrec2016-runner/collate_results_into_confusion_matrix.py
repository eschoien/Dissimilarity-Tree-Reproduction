import os
import json
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import sys

originalDir32x32 = '/mnt/VESSEL/main/Papers/202102 Paper 6 - Partial RICI Retrieval/Figure 12 - shrec2016-derived-matching-performance/simplesearch_32x32_original/output'
remeshedDir32x32 = '/mnt/VESSEL/main/Papers/202102 Paper 6 - Partial RICI Retrieval/Figure 12 - shrec2016-derived-matching-performance/simplesearch_32x32_remeshed/output'

originalDir64x64 = '/mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/derived/original_whamming'
remeshedDir64x64 = '/mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/derived/remeshed_whamming'

originalDir96x96 = '/mnt/VESSEL/main/Papers/202102 Paper 6 - Partial RICI Retrieval/Figure 12 - shrec2016-derived-matching-performance/simplesearch_96x96_original/output'
remeshedDir96x96 = '/mnt/VESSEL/main/Papers/202102 Paper 6 - Partial RICI Retrieval/Figure 12 - shrec2016-derived-matching-performance/simplesearch_96x96_remeshed/output'

originalDir = originalDir32x32
remeshedDir = remeshedDir32x32

fileCount = 383



def computeConfusionMatrix(inputDir):
    confusionMatrix = np.zeros((fileCount, fileCount))
    for fileIndex, file in enumerate(os.listdir(inputDir)):
        print(file)
        # filter out number from file name (example: T50.json -> 50)
        fileID = int(file[1:-5])
        with open(os.path.join(inputDir, file), 'r') as inputFile:
            fileContents = json.loads(inputFile.read())

        for i in range(0, fileCount):
            resultName = fileContents['results'][i]['name']
            resultScore = fileContents['results'][i]['score']
            # Also filter out file number, though this one will have a .dat extension instead
            resultIndex = int(resultName[1:-4])

            # register score
            confusionMatrix[fileID - 1, resultIndex - 1] = resultScore

        rowMin = min(confusionMatrix[fileID - 1])
        for i in range(0, fileCount):
            confusionMatrix[fileID - 1, i] -= rowMin
        rowMax = max(confusionMatrix[fileID - 1])
        for i in range(0, fileCount):
            confusionMatrix[fileID - 1, i] /= rowMax
    return confusionMatrix


originalMatrix = computeConfusionMatrix(originalDir)
remeshedMatrix = computeConfusionMatrix(remeshedDir)

globalMin = min(originalMatrix.min(), remeshedMatrix.min())
globalMax = max(originalMatrix.max(), remeshedMatrix.max())

print()
print('Confusion matrix:')


heatmap = plt.imshow(originalMatrix, cmap='hot', interpolation='nearest', vmin=globalMin, vmax=globalMax)

plt.figure(2)
plt.imshow(remeshedMatrix, cmap='hot', interpolation='nearest', vmin=globalMin, vmax=globalMax)
plt.colorbar(heatmap)#, format=ticker.FuncFormatter(fmt))
plt.show()