import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("resultsDirectory", help="Directory for reading results")
args = parser.parse_args()

#machine = "TURBONINJA"

#if machine == "TURBONINJA":
#	originalDir = '/mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/derived/original_whamming'
#	remeshedDir = '/mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/derived/remeshed_whamming'
#else:
#	originalDir = '/mnt/WAREHOUSE2/Stash/SHREC2016/results/derived/original'
#	remeshedDir = '/mnt/WAREHOUSE2/Stash/SHREC2016/results/derived/remeshed'

#inputDir = remeshedDir

fileCount = 383
histogram = [0] * fileCount

def processDirectory(inputDir):
    print('Processing directory:', inputDir)
    print('Found', len(os.listdir(inputDir)), 'result files')

    for file in os.listdir(inputDir):
        with open(os.path.join(inputDir, file), 'r') as inputFile:
            fileContents = json.loads(inputFile.read())
        fileToFind = file.replace('.json', '.dat')
        for i in range(0, fileCount):
            if fileToFind == fileContents['results'][i]['name']:
                histogram[i] += 1
                break
    print()
    print('Number of times the correct file appeared at search result n:')

    for i in range(0, fileCount):
        #if histogram[i] != 0:
        print(i + 1, '-', histogram[i])

processDirectory(args.resultsDirectory)