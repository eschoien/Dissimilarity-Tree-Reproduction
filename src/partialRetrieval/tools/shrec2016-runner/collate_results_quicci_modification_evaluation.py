import os
import os.path
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("sourceFilePath", help="JSON file produced by running the 'edgeRemovalExperiment' tool")
parser.add_argument("outputDirectory", help="Directory where output files should be written to")
args = parser.parse_args()

with open(args.sourceFilePath, 'r') as inFile:
    fileContents = json.loads(inFile.read())

with open(os.path.join(args.outputDirectory, 'unwanted_bit_reductions.csv'), 'w') as outFile:
    outFile.write('Reduction, Count\n')
    for entry in fileContents['reductionInWrongBitCounts']:
        outFile.write(str(entry['min']) + ", " + str(entry['count']) + '\n')

with open(os.path.join(args.outputDirectory, 'wrong_bit_counts_original.csv'), 'w') as outFile:
    outFile.write('Number of wrong bits, Count\n')
    for entry in fileContents['originalWrongBitCounts']:
        outFile.write(str(entry['min']) + ", " + str(entry['count']) + '\n')

with open(os.path.join(args.outputDirectory, 'wrong_bit_counts_modified.csv'), 'w') as outFile:
    outFile.write('Number of wrong bits, Count\n')
    for entry in fileContents['modifiedWrongBitCounts']:
        outFile.write(str(entry['min']) + ", " + str(entry['count']) + '\n')

with open(os.path.join(args.outputDirectory, 'overlap_with_reference.csv'), 'w') as outFile:
    outFile.write('Number of overlapping bits, Original QUICCI, Modified QUICCI\n')
    for index, entry in enumerate(fileContents['originalOverlapWithReference']):
        outFile.write(str(entry['min']) + ", " + str(entry['count'])
                                        + ', ' + str(fileContents['modifiedOverlapWithReference'][index]['count']) + '\n')