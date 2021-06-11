import scipy.io
import json
from pathlib import Path

mat = scipy.io.loadmat('SHREC2013/datasetSHREC2013_data.mat')

resultsFile = '/mnt/VESSEL/main/Papers/202102 Paper 6 - Partial RICI Retrieval/shrec2013-results/shrec2013_objectSearch_25.json'

with open(resultsFile, 'r') as resultsFileHandle:
    resultsFileContents = json.loads(resultsFileHandle.read())

queryInfo = []
targetInfo = []

print()
print('Reading Query info..')
for entry in mat['query'][0]:
    queryEntry = {}
    queryEntry['nameView'] = entry[0][0]
    queryEntry['object'] = entry[1][0]
    queryEntry['class'] = entry[2][0][0]
    queryEntry['numView'] = entry[3][0][0]
    queryEntry['path'] = entry[4][0]
    queryEntry['className'] = entry[5][0]
    queryEntry['index'] = entry[6][0][0]
    queryEntry['indexTarget'] = entry[7][0][0]
    queryInfo.append(queryEntry)

print()
print('Reading Target info..')
for entry in mat['target'][0]:
    targetEntry = {}
    targetEntry['object'] = entry[0][0]
    targetEntry['path'] = entry[1][0]
    targetEntry['class'] = entry[2][0][0]
    targetEntry['className'] = entry[3][0]
    targetEntry['index'] = entry[4][0][0]
    targetInfo.append(targetEntry)

correctCount = 0
for i in range(0, len(queryInfo)):
    resultsEntryFileID = int(Path(resultsFileContents['results'][i]['queryFile']).stem[1:]) - 1
    queryEntry = queryInfo[resultsEntryFileID]
    bestSearchResultFileID = int(Path(resultsFileContents['results'][i]['searchResults'][0]['objectFilePath']).stem[1:])
    print(queryEntry['index'], ':', queryEntry['indexTarget'], 'vs', bestSearchResultFileID)
    if queryEntry['indexTarget'] == bestSearchResultFileID:
        correctCount += 1

print('Correct count:', correctCount)
