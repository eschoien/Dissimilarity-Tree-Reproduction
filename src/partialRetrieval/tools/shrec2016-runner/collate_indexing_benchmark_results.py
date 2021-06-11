import os
import json

machine = "TURBONINJA"

if machine == "TURBONINJA":
    originalResultsFile = '/mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/indexed-search-benchmark/indexed_search_383_original_100000.txt'
    originalResultsFile = '/mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/indexed-search-benchmark/sequential_search_383_original_2500.txt'
    remeshedResultsFile = '/mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/derived/remeshed'
else:
    originalResultsFile = '/mnt/WAREHOUSE2/Stash/SHREC2016/results/derived/original'
    remeshedResultsFile = '/mnt/WAREHOUSE2/Stash/SHREC2016/results/derived/remeshed'

inputFile = originalResultsFile

factor = 10
histogramBins = 25

histogram = [0] * histogramBins * factor

averagesSumHistogram = [0] * histogramBins * factor
averagesCountsHistogram = [0.0000001] * histogramBins * factor
averagesMinHistogram = [9999999999] * histogramBins * factor
averagesMaxHistogram = [0] * histogramBins * factor


correctResultCount = 0

with open(inputFile, 'r') as inputFile:
    fileContents = json.loads(inputFile.read())
for entry in fileContents['results']:
    histogram[int(entry['executionTimeSeconds'] * factor)] += 1

    histogramBin = int(entry['executionTimeSeconds'] * factor)
    averagesSumHistogram[histogramBin] += entry['nodesVisited']
    averagesCountsHistogram[histogramBin] += 1
    averagesMaxHistogram[histogramBin] = max(entry['nodesVisited'], averagesMaxHistogram[histogramBin])
    averagesMinHistogram[histogramBin] = min(entry['nodesVisited'], averagesMinHistogram[histogramBin])

    if entry['queryFileID'] == entry['bestSearchResultFileID']:
        correctResultCount += 1
    print(entry['executionTimeSeconds'], ',', entry['bestMatchScore'], ',', entry['nodesVisited'])
print()


for i in range(0, histogramBins * factor):
    print(i / factor, '-', (i + 1) / factor, ',', histogram[i],
          ',', float(averagesSumHistogram[i]) / averagesCountsHistogram[i],
          ',', averagesMinHistogram[i],
          ',', averagesMaxHistogram[i])
print('Correct count:', correctResultCount / 1000)