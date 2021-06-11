import argparse
import csv
import os.path
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("resultsFile", help="File for reading results")
#parser.add_argument("outFile", help="output file")
args = parser.parse_args()

correctCount = 0

maxTimeSeconds = 600
histogramPrecision = 1
timeHistogram = (histogramPrecision * maxTimeSeconds) * [0]

with open(args.resultsFile, 'r') as inFile:
    fileContents = json.loads(inFile.read())

    for queryIndex, result in enumerate(fileContents['results']):
        if result['searchResults'][0]['objectID'] == queryIndex:
            correctCount += 1
        executionTime = result['executionTimeSeconds']
        print(executionTime)
        timeHistogramIndex = int(executionTime * histogramPrecision)
        timeHistogram[timeHistogramIndex] += 1

print('Correct count:', correctCount)
print('Execution time distribution:')
for i in range(0, maxTimeSeconds * histogramPrecision):
    print(float(i) / histogramPrecision, ',', timeHistogram[i])