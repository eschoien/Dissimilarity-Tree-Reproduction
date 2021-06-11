import os
import os.path
import json

sourceFilePath = '/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/edge_removal_verification/output.json'
outputDirectory = '/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/edge_removal_verification/'

with open(sourceFilePath, 'r') as inFile:
    fileContents = json.loads(inFile.read())

with open(os.path.join(outputDirectory, 'unwanted_bit_reductions.csv'), 'w') as outFile:
    outFile.write('Reduction, Count\n')
    for entry in fileContents['reductionInWrongBitCounts']:
        outFile.write(str(entry['min']) + ", " + str(entry['count']) + '\n')

with open(os.path.join(outputDirectory, 'wrong_bit_counts_original.csv'), 'w') as outFile:
    outFile.write('Number of wrong bits, Count\n')
    for entry in fileContents['originalWrongBitCounts']:
        outFile.write(str(entry['min']) + ", " + str(entry['count']) + '\n')

with open(os.path.join(outputDirectory, 'wrong_bit_counts_modified.csv'), 'w') as outFile:
    outFile.write('Number of wrong bits, Count\n')
    for entry in fileContents['modifiedWrongBitCounts']:
        outFile.write(str(entry['min']) + ", " + str(entry['count']) + '\n')

with open(os.path.join(outputDirectory, 'overlap_with_reference_original.csv'), 'w') as outFile:
    outFile.write('Number of overlapping bits, Count\n')
    for entry in fileContents['originalOverlapWithReference']:
        outFile.write(str(entry['min']) + ", " + str(entry['count']) + '\n')

with open(os.path.join(outputDirectory, 'overlap_with_reference_modified.csv'), 'w') as outFile:
    outFile.write('Number of overlapping bits, Count\n')
    for entry in fileContents['modifiedOverlapWithReference']:
        outFile.write(str(entry['min']) + ", " + str(entry['count']) + '\n')