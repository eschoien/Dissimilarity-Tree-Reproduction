import argparse
import csv
import os.path
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("resultsFile", help="File for reading results")
parser.add_argument("outFile", help="output file")
args = parser.parse_args()



with open(args.resultsFile, 'r') as inFile:
    fileContents = json.loads(inFile.read())

    normalisedSums = [0] * fileContents['resultCount']
    averages = [0] * fileContents['resultCount']

    for resultSet in fileContents['results']:
        setScores = [entry['score'] for entry in resultSet['searchResultFileIDs']]
        bestScore = max(setScores[0], 1)
        normalisedScores = [score / bestScore for score in setScores]
        for i in range(0, len(normalisedScores)):
            normalisedSums[i] += normalisedScores[i]

    for i in range(0, fileContents['resultCount']):
        averages[i] = normalisedSums[i] / float(fileContents['sampleCount'])

    with open(args.outFile, 'w') as outFile:
        stringScores = [str(score) for score in averages]
        outFile.write(', '.join(stringScores))
        outFile.write('\n')