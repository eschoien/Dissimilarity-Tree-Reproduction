import json
import os
import shutil
import subprocess
import random
import sys
import multiprocessing
import hashlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.simple_term_menu import TerminalMenu

from scripts.prettytable import PrettyTable

# Some global variables to hold on to settings and constants that are used everywhere
gpuID = 0
mainEvaluationRandomSeed = '725948161'
shrec2016_support_radius = '100'
descriptorWidthBits = 64
indexGenerationMode = 'GPU'
pipelineEvaluation_queryMode = 'Best Case'
pipelineEvaluation_consensusThreshold = '10'
pipelineEvaluation_resolution = '32x32'

if not (sys.version_info.major == 3 and sys.version_info.minor >= 8):
    print("This script requires Python 3.8 or higher.")
    print("You are using Python {}.{}.".format(sys.version_info.major, sys.version_info.minor))
    sys.exit(1)

def run_command_line_command(command, working_directory='.'):
    print('>> Executing command:', command)
    subprocess.run(command, shell=True, check=False, cwd=working_directory)

def ask_for_confirmation(message):
    confirmation_menu = TerminalMenu(["yes", "no"], title=message)
    choice = confirmation_menu.show()
    return choice == 0

def downloadFile(fileURL, tempFile, extractInDirectory, name):
    if not os.path.isfile('input/download/' + tempFile) or ask_for_confirmation('It appears the ' + name + ' archive file has already been downloaded. Would you like to download it again?'):
        print('Downloading the ' + name + ' archive file..')
        run_command_line_command('wget --output-document ' + tempFile + ' ' + fileURL, 'input/download/')
    print()
    os.makedirs(extractInDirectory, exist_ok=True)
    run_command_line_command('p7zip -k -d ' + os.path.join(os.path.relpath('input/download', extractInDirectory), tempFile), extractInDirectory)
    #if ask_for_confirmation('Download and extraction complete. Would you like to delete the compressed archive to save disk space?'):
    #    os.remove('input/download/' + tempFile)
    print()

# SHREC2016: https://ntnu.box.com/shared/static/zb2co430vdcpao7gwco3vaxsf7ahz09u.7z (1.2GB uncompressed, 258MB compressed)
# Index 64x64: https://ntnu.box.com/shared/static/cv4h14yqy9tx5llyc4t2tbbpr1nak52r.7z (22GB uncompressed, 3.6GB compressed)
# Index 96x96: https://ntnu.box.com/shared/static/q1blnwzrq8g0cuh3pl3f0av3v4n4qqi6.7z (48GB uncompressed, 7.2GB compressed)
# Index 32x32: https://ntnu.box.com/shared/static/g5ckpzuqilcx2oknvytbh8l4uyv08ifv.7z (5.6GB uncompressed, 1.1GB compressed)
# Augmented dataset: https://ntnu.box.com/shared/static/e57v52moxf3g0fx394cs7bhfo6oto4mr.7z (720MB uncompressed, 190MB compressed)

def downloadDatasetsMenu():
    download_menu = TerminalMenu([
        "Download all",
        "Download SHREC 2016 partial 3D shape dataset ()",
        #'Download augmented SHREC\'16 query dataset',
        'Download precomputed descriptors',
        "Download precomputed dissimilarity tree indexes",
        "back"], title='------------------ Download Datasets ------------------')

    while True:
        choice = download_menu.show() + 1
        os.makedirs('input/download/', exist_ok=True)

        if choice == 1 or choice == 2:
           downloadFile('https://ntnu.box.com/shared/static/zb2co430vdcpao7gwco3vaxsf7ahz09u.7z', 'SHREC2016.7z',
                        'input/SHREC2016_partial_retrieval', 'SHREC 2016 Partial Retrieval Dataset')

        if choice == 1 or choice == 3:
            pass
            #downloadFile('https://ntnu.box.com/shared/static/e57v52moxf3g0fx394cs7bhfo6oto4mr.7z', 'SHREC2016_augmented.7z',
            #             'input/precomputed_augmented_dataset', 'Augmented SHREC 2016 Query Dataset')

        if choice == 1 or choice == 4:
            downloadFile('https://ntnu.box.com/shared/static/q1blnwzrq8g0cuh3pl3f0av3v4n4qqi6.7z', 'index_96x96.7z',
                         'input/precomputed_dissimilarity_trees/index_96x96', 'Precomputed Dissimilarity Tree for Descriptors of resolution 96x96')
            downloadFile('https://ntnu.box.com/shared/static/cv4h14yqy9tx5llyc4t2tbbpr1nak52r.7z', 'index_64x64.7z',
                         'input/precomputed_dissimilarity_trees/index_64x64', 'Precomputed Dissimilarity Tree for Descriptors of resolution 64x64')
            downloadFile('https://ntnu.box.com/shared/static/g5ckpzuqilcx2oknvytbh8l4uyv08ifv.7z', 'index_32x32.7z',
                         'input/precomputed_dissimilarity_trees/index_32x32', 'Precomputed Dissimilarity Tree for Descriptors of resolution 32x32')
        if choice == 6:
            return

def installDependenciesMenu():
    install_menu = TerminalMenu([
        "Install all dependencies except CUDA",
        "Install CUDA (through APT)",
        "back"], title='---------------- Install Dependencies ----------------')

    while True:
        choice = install_menu.show()

        if choice == 0:
            run_command_line_command('sudo apt install cmake python3 python3-pip libpcl-dev g++ gcc build-essential wget p7zip')
            run_command_line_command('sudo pip3 install simple-term-menu xlwt xlrd numpy matplotlib pillow PyQt5')
            print()
        if choice == 1:
            run_command_line_command('sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev')
            print()
        if choice == 2:
            return

def changeDescriptorWidth(newWidth):
    run_command_line_command("sed -i 's/^#define spinImageWidthPixels .*/#define spinImageWidthPixels " + str(newWidth) + "/' src/libShapeDescriptor/src/shapeDescriptor/libraryBuildSettings.h")

def compileProject():
    print('This project uses cmake for generating its makefiles.')
    print('It has a tendency to at times be unable to find an installed CUDA compiler.')
    print('Also, depending on which version of CUDA you have installed, you may need')
    print('to change the version of GCC/G++ used for compatibility reasons.')
    print('If either of these occurs, modify the paths at the top of the following file: ')
    print('    src/partialRetrieval/CMakeLists.txt')
    print()

    run_command_line_command('rm -rf bin/*')

    os.makedirs('bin/build32x32', exist_ok=True)
    os.makedirs('bin/build64x64', exist_ok=True)
    os.makedirs('bin/build96x96', exist_ok=True)

    threadCount = str(multiprocessing.cpu_count() - 1)

    changeDescriptorWidth(32)
    run_command_line_command('cmake ../../src/partialRetrieval -DCMAKE_BUILD_TYPE=Release', 'bin/build32x32')
    run_command_line_command('make -j ' + threadCount, 'bin/build32x32')

    changeDescriptorWidth(64)
    run_command_line_command('cmake ../../src/partialRetrieval -DCMAKE_BUILD_TYPE=Release', 'bin/build64x64')
    run_command_line_command('make -j ' + threadCount, 'bin/build64x64')

    changeDescriptorWidth(96)
    run_command_line_command('cmake ../../src/partialRetrieval -DCMAKE_BUILD_TYPE=Release', 'bin/build96x96')
    run_command_line_command('make -j ' + threadCount, 'bin/build96x96')

    print()
    print('Complete.')
    print()

def printAvailableGPUs():
    run_command_line_command('bin/build32x32/descriptorDumper --list-gpus')

def configureGPU():
    global gpuID
    printAvailableGPUs()
    print()
    gpuID = input('Enter the ID of the GPU to use (usually 0): ')
    print()

def fileMD5(filePath):
    with open(filePath, 'rb') as inFile:
        return hashlib.md5(inFile.read()).hexdigest()

def generateAugmentedDataset():
    os.makedirs('output/augmented_dataset_original', exist_ok=True)
    os.makedirs('output/augmented_dataset_remeshed', exist_ok=True)

    run_command_line_command('bin/build32x32/querysetgenerator '
                             '--object-directory=input/SHREC2016_partial_retrieval/complete_objects '
                             '--output-directory=output/augmented_dataset_original '
                             '--random-seed=' + mainEvaluationRandomSeed)
    run_command_line_command('bin/build32x32/querysetgenerator '
                             '--object-directory=input/SHREC2016_partial_retrieval/complete_objects '
                             '--output-directory=output/augmented_dataset_remeshed '
                             '--redistribute-triangles '
                             '--random-seed=' + mainEvaluationRandomSeed)

def computeDescriptorsFromFile(inputFile, outputFile, descriptorWidth):
    run_command_line_command('bin/build' + descriptorWidth + 'x' + descriptorWidth + '/descriptorDumper'
                   + ' --input-file="' + inputFile
                   + '" --output-file="' + outputFile
                   + '" --support-radius=' + str(shrec2016_support_radius))

def computeDescriptorsFromDirectory(inputDirectory, outputDirectory, descriptorWidth):
    os.makedirs(outputDirectory, exist_ok=True)
    filesToProcess = [f for f in os.listdir(inputDirectory) if os.path.isfile(os.path.join(inputDirectory, f))]
    print('Computing images: ', inputDirectory, '->', outputDirectory)
    print('Found', len(filesToProcess), 'files')
    for index, fileToProcess in enumerate(filesToProcess):
        inputFilePath = os.path.join(inputDirectory, fileToProcess)
        dumpFilePath = os.path.join(outputDirectory, fileToProcess[0:-4] + '.dat')
        print('\tProcessing file', (index + 1), '/', len(filesToProcess), ':', fileToProcess)

        computeDescriptorsFromFile(inputFilePath, dumpFilePath, descriptorWidth)


descriptorInputDirectories = ['input/SHREC2016_partial_retrieval/complete_objects',
                              'output/augmented_dataset_original',
                              'output/augmented_dataset_remeshed',
                              'input/SHREC2016_partial_retrieval/queries_artificial/Q25',
                              'input/SHREC2016_partial_retrieval/queries_artificial/Q40']

def getDescriptorDirectoriesByResolution(resolution):
    outputDirectories = ['output/descriptors/complete_objects_' + resolution + 'x' + resolution,
                         'output/descriptors/augmented_dataset_original_' + resolution + 'x' + resolution,
                         'output/descriptors/augmented_dataset_remeshed_' + resolution + 'x' + resolution,
                         'output/descriptors/shrec2016_25partiality_' + resolution + 'x' + resolution,
                         'output/descriptors/shrec2016_40partiality_' + resolution + 'x' + resolution]
    return outputDirectories

def computeDescriptorBatch(batchIndex, resolution):
    outputDirectories = getDescriptorDirectoriesByResolution(resolution)
    computeDescriptorsFromDirectory(descriptorInputDirectories[batchIndex], outputDirectories[batchIndex], resolution)

def computeDescriptors():
    for descriptorwidth in ['32', '64', '96']:
        os.makedirs('output/descriptors/complete_objects_' + descriptorwidth + 'x' + descriptorwidth, exist_ok=True)
        os.makedirs('output/descriptors/augmented_dataset_original_' + descriptorwidth + 'x' + descriptorwidth, exist_ok=True)
        os.makedirs('output/descriptors/augmented_dataset_remeshed_' + descriptorwidth + 'x' + descriptorwidth, exist_ok=True)
        os.makedirs('output/descriptors/shrec2016_25partiality_' + descriptorwidth + 'x' + descriptorwidth, exist_ok=True)
        os.makedirs('output/descriptors/shrec2016_40partiality_' + descriptorwidth + 'x' + descriptorwidth, exist_ok=True)

    while True:
        run_menu = TerminalMenu([
            "Generate all descriptors (will take several hours)",
            "Copy all descriptors precomputed by authors",
            "Generate descriptors for one random object from each set and verify against authors",
            "back"], title='-- Compute descriptors --')
        choice = run_menu.show() + 1
        if choice == 1:
            for index, descriptorwidth in enumerate(['32', '64', '96']):
                changeDescriptorWidth(int(descriptorwidth))
                print('Processing batch 1/5 in resolution ' + str(index + 1) + '/3 (' + descriptorwidth + 'x' + descriptorwidth + ')')
                computeDescriptorBatch(0, descriptorwidth)
                print('Processing batch 2/5 in resolution ' + str(index + 1) + '/3 (' + descriptorwidth + 'x' + descriptorwidth + ')')
                computeDescriptorBatch(1, descriptorwidth)
                print('Processing batch 3/5 in resolution ' + str(index + 1) + '/3 (' + descriptorwidth + 'x' + descriptorwidth + ')')
                computeDescriptorBatch(2, descriptorwidth)
                print('Processing batch 4/5 in resolution ' + str(index + 1) + '/3 (' + descriptorwidth + 'x' + descriptorwidth + ')')
                computeDescriptorBatch(3, descriptorwidth)
                print('Processing batch 5/5 in resolution ' + str(index + 1) + '/3 (' + descriptorwidth + 'x' + descriptorwidth + ')')
                computeDescriptorBatch(4, descriptorwidth)
        if choice == 2:
            print('Copying precomputed descriptors..')
            shutil.copytree('input/descriptors', 'output/descriptors', dirs_exist_ok=True)
        if choice == 3:
            os.makedirs('output/descriptors/temp', exist_ok=True)
            for index, descriptorwidth in enumerate(['32', '64', '96']):
                for directoryIndex, inputDirectory in enumerate(descriptorInputDirectories):
                    chosenFile = random.choice(os.listdir(inputDirectory))
                    print('Processing directory', str(directoryIndex+1) + '/' + str(len(descriptorInputDirectories)),
                          'at resolution', str(index+1) + '/3 (' + descriptorwidth + 'x' + descriptorwidth + '):',
                          'chose file', chosenFile)
                    computeDescriptorsFromFile(os.path.join(inputDirectory, chosenFile),
                                               'output/descriptors/temp/descriptors.dat', descriptorwidth)
                    computedHash = fileMD5('output/descriptors/temp/descriptors.dat')

                    outputDirectories = getDescriptorDirectoriesByResolution(descriptorwidth)
                    referenceHash = fileMD5(os.path.join(outputDirectories[directoryIndex].replace('output', 'input'), chosenFile.replace('.obj', '.dat')))

                    print('  File hashes - computed:', computedHash, 'authors:', referenceHash,
                          '- MATCHES' if computedHash == referenceHash else "- !!! DOES NOT MATCH !!!")
        if choice == 4:
            return

def configureIndexGeneration():
    global indexGenerationMode

    cpu_or_gpu_menu = TerminalMenu([
        "Use CPU for index construction",
        "Use GPU for index construction"], title='---------------- Configure CPU or GPU indexing ----------------')

    choice = cpu_or_gpu_menu.show()

    if choice == 0:
        indexGenerationMode = 'CPU'
    if choice == 1:
        indexGenerationMode = 'GPU'

def computeDissimilarityTree():
    os.makedirs('output/dissimilarity_tree/index32x32', exist_ok=True)
    os.makedirs('output/dissimilarity_tree/index64x64', exist_ok=True)
    os.makedirs('output/dissimilarity_tree/index96x96', exist_ok=True)

    while True:
        run_menu = TerminalMenu([
            "Configure CPU or GPU based index generation (current: " + indexGenerationMode + ')',
            "Compute index for 32x32 descriptors",
            "Compute index for 64x64 descriptors",
            "Compute index for 96x96 descriptors",
            "Copy precomputed index of 32x32 descriptors",
            "Copy precomputed index of 64x64 descriptors",
            "Copy precomputed index of 96x96 descriptors",
            "back"], title='-- Compute descriptors --')

        choice = run_menu.show() + 1

        indexGenerationCommand = 'clusterbuilder ' \
                                 '--quicci-dump-directory="input/SHREC2016_partial_retrieval/complete_objects" ' \
                                 '--force-gpu=' + str(gpuID) + ' ' \
                                 '--force-cpu=' + ('true' if indexGenerationMode == 'CPU' else 'false')

        if choice == 1:
            configureIndexGeneration()
        if choice == 2:
            run_command_line_command('bin/build32x32/' + indexGenerationCommand
                                     + ' --index-directory=output/dissimilarity_tree/index32x32'
                                       ' --quicci-dump-directory=output/descriptors/complete_objects_32x32')
        if choice == 3:
            run_command_line_command('bin/build64x64/' + indexGenerationCommand
                                     + ' --index-directory=output/dissimilarity_tree/index64x64'
                                       ' --quicci-dump-directory=output/descriptors/complete_objects_64x64')
        if choice == 4:
            run_command_line_command('bin/build96x96/' + indexGenerationCommand
                                     + ' --index-directory=output/dissimilarity_tree/index96x96'
                                       ' --quicci-dump-directory=output/descriptors/complete_objects_96x96')
        if choice == 5:
            print('Copying precomputed index of 32x32 images..')
            shutil.copy('input/precomputed_dissimilarity_trees/index_32x32/index.dat', 'output/dissimilarity_tree/index32x32/index.dat')
        if choice == 6:
            print('Copying precomputed index of 64x64 images..')
            shutil.copy('input/precomputed_dissimilarity_trees/index_64x64/index.dat', 'output/dissimilarity_tree/index64x64/index.dat')
        if choice == 7:
            print('Copying precomputed index of 96x96 images..')
            shutil.copy('input/precomputed_dissimilarity_trees/index_96x96/index.dat', 'output/dissimilarity_tree/index96x96/index.dat')
        if choice == 8:
            return

def runVoteCountProgressionExperiment():
    os.makedirs('output/Figure_3_voteCountProgression/input', exist_ok=True)
    shutil.copy('output/augmented_dataset_remeshed/T103.obj',
                'output/Figure_3_voteCountProgression/input/T103.obj')
    run_command_line_command('bin/build64x64/objectSearch '
                             '--index-directory=output/dissimilarity_tree/index64x64 '
                             '--haystack-directory=input/SHREC2016_partial_retrieval/complete_objects '
                             '--query-directory=output/Figure_3_voteCountProgression/input '
                             '--resultsPerQueryImage=1 '
                             '--randomSeed=' + mainEvaluationRandomSeed + ' '
                             '--support-radius=' + shrec2016_support_radius + ' '
                             '--consensus-threshold=1000 '
                             '--force-gpu=' + str(gpuID) + ' '
                             '--output-progression-file=output/Figure_3_voteCountProgression/query_progression.csv '
                             '--progression-iteration-limit=1000')

    print()
    print('Done! You can now open the file output/Figure_3_voteCountProgression/input/T103.obj, and create a chart of all columns.')
    print('It should exactly match the one shown in the paper.')
    print()

def computeAverageScoreChart():
    os.makedirs('output/Figure_4_averageRelativeDistance', exist_ok=True)

    while True:
        run_menu = TerminalMenu([
            "Compute random search result and compare it to the one computed by authors",
            "Compute chart based on search results computed by authors",
            "Compute all search results, then compute chart",
            "back"], title='-- Reproduce Figure 4: average relative distance chart --')

        choice = run_menu.show() + 1
        resultsFileToProcess = ''
        runCommand = 'bin/build64x64/indexedSearchBenchmark ' \
                     '--index-directory=output/dissimilarity_tree/index64x64 ' \
                     '--query-directory=output/augmented_dataset_original ' \
                     '--output-file=output/Figure_4_averageRelativeDistance/measurements.json ' \
                     '--search-results-per-query=50 ' \
                     '--random-seed=' + mainEvaluationRandomSeed + ' ' \
                     '--support-radius=' + shrec2016_support_radius + ' ' \
                     '--sample-count=1000 ' \
                     '--force-gpu=' + str(gpuID) + ' '

        if choice == 1:
            objectIndexToTest = random.randint(0, 1000)
            print()
            print('Randomly selected query image #', (objectIndexToTest + 1))
            print()
            run_command_line_command(runCommand + ' --single-query-index=' + str(objectIndexToTest))
            print()
            with open('output/Figure_4_averageRelativeDistance/measurements.json', 'r') as inFile:
                computedResults = json.loads(inFile.read())
            with open('input/precomputed_results/figure4_results_authors.json', 'r') as inFile:
                referenceResults = json.loads(inFile.read())

            outputTable = PrettyTable(['Score (computed)', 'File ID (computed)', 'Image ID (computed)', '', 'Score (authors)', 'File ID (authors)', 'Image ID (authors)'])
            outputTable.align = "l"
            for i in range(0, 50):
                outputTable.add_row([computedResults['results'][0]['searchResultFileIDs'][i]['score'],
                                     computedResults['results'][0]['searchResultFileIDs'][i]['fileID'],
                                     computedResults['results'][0]['searchResultFileIDs'][i]['imageID'],
                                     '',
                                     referenceResults['results'][objectIndexToTest]['searchResultFileIDs'][i]['score'],
                                     referenceResults['results'][objectIndexToTest]['searchResultFileIDs'][i]['fileID'],
                                    referenceResults['results'][objectIndexToTest]['searchResultFileIDs'][i]['imageID']])
            print(outputTable)

        if choice == 2:
            resultsFileToProcess = 'input/precomputed_results/figure4_results_authors.json'
        if choice == 3:
            run_command_line_command(runCommand)
            resultsFileToProcess = 'output/Figure_4_averageRelativeDistance/measurements.json '
        if choice == 2 or choice == 3:
            run_command_line_command('python3 src/partialRetrieval/tools/shrec2016-runner/compute_relative_score_chart.py '
                                     + resultsFileToProcess +
                                     ' output/Figure_4_averageRelativeDistance/chart_values.csv')
            print()
            print('Complete. The computed chart values have been written to:')
            print('output/Figure_4_averageRelativeDistance/chart_values.csv')
            print()
        if choice == 4:
            return

def computeBitsHeatmap():
    os.makedirs('output/Figure_6_OccurrenceCountHeatmap', exist_ok=True)
    run_command_line_command('bin/build64x64/occurrenceCounter '
                             '--index-directory=output/dissimilarity_tree/index64x64 '
                             '--output-file=output/Figure_6_OccurrenceCountHeatmap/shrec16_occurrence_counts.txt')
    run_command_line_command('python3 src/partialRetrieval/tools/shrec2016-runner/heatmap.py '
                             'output/Figure_6_OccurrenceCountHeatmap/shrec16_occurrence_counts.txt')

def collateIndexEvaluationResults(indexedInputFile, sequentialInputFile, outputFile):
    factor = 10
    histogramBins = 150

    def computeHistograms(inputFile):
        histogram = [0] * histogramBins * factor

        averagesSumHistogram = [0] * histogramBins * factor
        averagesCountsHistogram = [0.0000001] * histogramBins * factor
        averagesMinHistogram = [9999999999] * histogramBins * factor
        averagesMaxHistogram = [0] * histogramBins * factor

        with open(inputFile, 'r') as inputFile:
            fileContents = json.loads(inputFile.read())
        for entry in fileContents['results']:
            histogram[int(entry['executionTimeSeconds'] * factor)] += 1

            histogramBin = int(entry['executionTimeSeconds'] * factor)
            averagesSumHistogram[histogramBin] += entry['nodesVisited']
            averagesCountsHistogram[histogramBin] += 1
            averagesMaxHistogram[histogramBin] = max(entry['nodesVisited'], averagesMaxHistogram[histogramBin])
            averagesMinHistogram[histogramBin] = min(entry['nodesVisited'], averagesMinHistogram[histogramBin])
        return (histogram, averagesSumHistogram, averagesCountsHistogram, averagesMinHistogram, averagesMaxHistogram)

    indexedHistogram, averagesSumHistogram, averagesCountsHistogram, averagesMinHistogram, averagesMaxHistogram = computeHistograms(indexedInputFile)
    sequentialHistogram, _, _, _, _ = computeHistograms(sequentialInputFile)

    with open(outputFile, 'w') as outFile:
        outFile.write('Execution Time (s), Number of Queries executed (Proposed), Number of Queries executed (Sequential), Number of Queries executed (Sequential and scaled by a factor of 10), ')
        outFile.write('Average Node Count Visited, Minimum Node Count Visited, Maximum Node Count Visited')
        outFile.write('\n')

        for i in range(0, histogramBins * factor):
            if indexedHistogram[i] == 0 and sequentialHistogram[i] == 0:
                # Skip time slices which contain no queries
                continue
            outFile.write(','.join(map(str, (i / factor, indexedHistogram[i], sequentialHistogram[i], 10 * sequentialHistogram[i]))))
            if indexedHistogram[i] != 0:
                outFile.write(',' + ','.join(map(str, (float(averagesSumHistogram[i]) / averagesCountsHistogram[i],
                                                       averagesMinHistogram[i], averagesMaxHistogram[i]))))
            else:
                outFile.write(', , , ')
            outFile.write('\n')

def runIndexEvaluation():
    os.makedirs('output/Figure_10_and_17_indexQueryTimes', exist_ok=True)

    baseIndexedSearchCommand = 'bin/build64x64/indexedSearchBenchmark ' \
                               '--index-directory=output/dissimilarity_tree/index64x64 ' \
                               '--query-directory=output/augmented_dataset_original ' \
                               '--output-file=output/Figure_10_and_17_indexQueryTimes/measurements_indexed.json ' \
                               '--search-results-per-query=1 ' \
                               '--random-seed=' + mainEvaluationRandomSeed + ' ' \
                               '--support-radius=' + shrec2016_support_radius + ' ' \
                               '--sample-count=100000 ' \
                               '--force-gpu=' + str(gpuID) + ' '

    baseSequentialSearchCommand = 'bin/build64x64/sequentialSearchBenchmark ' \
                                  '--index-directory=output/dissimilarity_tree/index64x64 ' \
                                  '--query-directory=output/augmented_dataset_original ' \
                                  '--random-seed=' + mainEvaluationRandomSeed + ' ' \
                                  '--output-file=output/Figure_10_and_17_indexQueryTimes/measurements_sequential.json ' \
                                  '--sample-count=2500 ' \
                                  '--force-gpu=' + str(gpuID) + ' '

    while True:
        run_menu = TerminalMenu([
            "Compute random batch of 50 indexed queries",
            "Compute random batch of 10 sequential searches",
            "Compute chart based on search results computed by authors",
            "Compute entire chart from scratch",
            "back"], title='-- Reproduce Figure 10: Dissimilarity Tree Query Times --')

        choice = run_menu.show() + 1

        if choice == 1:
            resultsToCompute = 50
            startIndex = random.randint(0, 100000 - resultsToCompute)
            run_command_line_command(baseIndexedSearchCommand +
                                     '--subset-start-index=' + str(startIndex) + ' '
                                     '--subset-end-index=' + str(startIndex + resultsToCompute))
            with open('output/Figure_10_and_17_indexQueryTimes/measurements_indexed.json', 'r') as inFile:
                computedResults = json.loads(inFile.read())
            with open('input/precomputed_results/figure10_indexed_search_100000.json', 'r') as inFile:
                chartExecutionTimes = json.loads(inFile.read())
            outputTable = PrettyTable(['Query ID', 'Execution Time (computed)', 'Best Match (computed)', '',
                                       'Execution Time (authors)', 'Best Match (authors)'])
            outputTable.align = "l"
            for i in range(0, resultsToCompute):
                outputTable.add_row([startIndex + i,
                                     computedResults['results'][i]['executionTimeSeconds'],
                                     'File ' + str(computedResults['results'][i]['bestSearchResultFileID']) + ', image '
                                     + str(computedResults['results'][i]['bestSearchResultImageID']), '',
                                     chartExecutionTimes['results'][startIndex + i]['executionTimeSeconds'],
                                     'File ' + str(chartExecutionTimes['results'][startIndex + i]['bestSearchResultFileID']) + ', image '
                                     + str(chartExecutionTimes['results'][startIndex + i]['bestSearchResultImageID'])])
            print(outputTable)

        if choice == 2:
            resultsToCompute = 10
            startIndex = random.randint(0, 2500 - resultsToCompute)
            run_command_line_command(baseSequentialSearchCommand +
                                     '--subset-start-index=' + str(startIndex) + ' '
                                     '--subset-end-index=' + str(startIndex + resultsToCompute))
            with open('output/Figure_10_and_17_indexQueryTimes/measurements_sequential.json', 'r') as inFile:
                computedResults = json.loads(inFile.read())
            with open('input/precomputed_results/figure10_sequential_search_2500.json', 'r') as inFile:
                chartExecutionTimes = json.loads(inFile.read())
            outputTable = PrettyTable(['Query ID', 'Execution Time (computed)', 'Best Match (computed)', '',
                                       'Execution Time (authors)', 'Best Match (authors)'])
            outputTable.align = "l"
            for i in range(0, resultsToCompute):
                outputTable.add_row([startIndex + i,
                                     computedResults['results'][i]['executionTimeSeconds'],
                                     'File ' + str(computedResults['results'][i]['bestSearchResultFileID']) + ', image '
                                     + str(computedResults['results'][i]['bestSearchResultImageID']), '',
                                     chartExecutionTimes['results'][startIndex + i]['executionTimeSeconds'],
                                     'File ' + str(chartExecutionTimes['results'][startIndex + i]['bestSearchResultFileID']) + ', image '
                                     + str(chartExecutionTimes['results'][startIndex + i]['bestSearchResultImageID'])])
            print(outputTable)

        if choice == 3:
            indexedResultsFile = 'input/precomputed_results/figure10_indexed_search_100000.json'
            sequentialResultsFile = 'input/precomputed_results/figure10_sequential_search_2500.json'
            outputFile = 'output/Figure_10_and_17_indexQueryTimes/authors_indexed_search_times.csv'
        if choice == 4:
            run_command_line_command(baseIndexedSearchCommand)
            run_command_line_command(baseSequentialSearchCommand)
            indexedResultsFile = 'output/Figure_10_and_17_indexQueryTimes/measurements_indexed.json'
            sequentialResultsFile = 'output/Figure_10_and_17_indexQueryTimes/measurements_sequential.json'
            outputFile = 'output/Figure_10_and_17_indexQueryTimes/computed_indexed_search_times.csv'

        if choice == 3 or choice == 4:
            print()
            print('Compiling results..')
            collateIndexEvaluationResults(indexedResultsFile, sequentialResultsFile, outputFile)
            print()
            print('Done. You can find the produced CSV file here:')
            print('    ' + outputFile)
            print()
            print('Use columns 0, 1, and 3 for Figure 10.')
            print('Use columns 0, 4, 5, and 6 for Figure 17.')
            print()

        if choice == 5:
            return


def runModifiedQuicciEvaluation():
    os.makedirs('output/Figure_11_and_12_unwantedBitEvaluation', exist_ok=True)

    run_command_line_command('bin/build64x64/edgeRemovalExperiment '
                             '--query-directory=output/augmented_dataset_original '
                             '--reference-object-directory=input/SHREC2016_partial_retrieval/complete_objects '
                             '--output-file=output/Figure_11_and_12_unwantedBitEvaluation/output.json '
                             '--force-gpu=' + str(gpuID) + ' '
                             '--support-radius=' + shrec2016_support_radius)

    run_command_line_command('python3 src/partialRetrieval/tools/shrec2016-runner/collate_results_quicci_modification_evaluation.py '
                             'output/Figure_11_and_12_unwantedBitEvaluation/output.json '
                             'output/Figure_11_and_12_unwantedBitEvaluation/')
    print()
    print('All done!')
    print('To create the figures shown in the paper:')
    print('Figure 11: Open output/Figure_11_and_12_unwantedBitEvaluation/unwanted_bit_reductions.csv')
    print('           Then create a chart from its contents.')
    print('Figure 12: Open output/Figure_11_and_12_unwantedBitEvaluation/overlap_with_reference.csv')
    print('           And create a chart from its contents.')
    print()

def computeAllToAllReferenceDirectory(remeshed, disableModifiedQUICCI):
    referenceFileBaseDirectory = 'input/precomputed_results/figure13_and_table1/'
    referenceDirectoryName = ('alltoall_remeshed_' if remeshed else 'alltoall_bestcase_') + \
                             ('originalquicci_64x64/' if disableModifiedQUICCI else 'modifiedquicci_64x64/')
    referenceDirectory = referenceFileBaseDirectory + referenceDirectoryName
    return referenceDirectory

def computeAllToAllReplicatedDirectory(remeshed, disableModifiedQUICCI):
    if remeshed:
        outputBasePathPart = 'output/Figure_13_and_Table_1_AllToAllSearch/results_augmentedremeshed_'
    else:
        outputBasePathPart = 'output/Figure_13_and_Table_1_AllToAllSearch/results_augmentedbestcase_'

    if disableModifiedQUICCI:
        outputBasePath = outputBasePathPart + 'originalquicci'
    else:
        outputBasePath = outputBasePathPart + 'modifiedquicci'

    return outputBasePath

def runSingleAllToAll(queryMesh, remeshed, disableModifiedQUICCI):
    outputBasePath = computeAllToAllReplicatedDirectory(remeshed, disableModifiedQUICCI)

    os.makedirs(outputBasePath, exist_ok=True)
    outputFile = os.path.join(outputBasePath, os.path.basename(queryMesh).replace('.obj', '.json'))

    run_command_line_command('bin/build64x64/simplesearch '
                             '--haystack-directory=output/descriptors/complete_objects_64x64 '
                             '--query-mesh=' + queryMesh + ' '
                             '--force-gpu=' + str(gpuID) + ' '
                             '--output-file=' + outputFile + ' ' +
                            ('--disable-modified-quicci' if disableModifiedQUICCI else ''))

    with open(outputFile, 'r') as inFile:
        replicatedFileContents = json.loads(inFile.read())

    referenceFilePath = computeAllToAllReferenceDirectory(remeshed, disableModifiedQUICCI) + os.path.basename(outputFile)

    with open(referenceFilePath, 'r') as inFile:
        referenceFileContents = json.loads(inFile.read())

    outputTable = PrettyTable(['Rank', 'Search Result (computed)', 'Execution Time (computed)', 'Summed Distance (computed)', '',
                               'Search Result (authors)', 'Execution Time (authors)', 'Summed Distance (authors)'])
    outputTable.align = "l"
    for i in range(0, len(replicatedFileContents['results'])):
        outputTable.add_row([i + 1, replicatedFileContents['results'][i]['name'],
                                    replicatedFileContents['results'][i]['executionTimeSeconds'],
                                    replicatedFileContents['results'][i]['score'], '',
                                    referenceFileContents['results'][i]['name'],
                                    referenceFileContents['results'][i]['executionTimeSeconds'],
                                    referenceFileContents['results'][i]['score']])
    print()
    print(outputTable)
    print()


def processAllToAllResultsDirectory(inputDir):
    with open(os.path.join(inputDir, os.listdir(inputDir)[0]), 'r') as inputFile:
        firstFileContents = json.loads(inputFile.read())
    resultCount = len(firstFileContents['results'])
    histogram = [0] * resultCount
    for file in os.listdir(inputDir):
        with open(os.path.join(inputDir, file), 'r') as inputFile:
            fileContents = json.loads(inputFile.read())
        fileToFind = file.replace('.json', '.dat')
        for i in range(0, resultCount):
            if fileToFind == fileContents['results'][i]['name']:
                histogram[i] += 1
                break

    return float(histogram[0]) / float(resultCount)

def computeHeatmaps(bestCaseResultsDirectory, remeshedResultsDirectory):
    fileCount = len(os.listdir(bestCaseResultsDirectory))

    def computeConfusionMatrix(inputDir):
        confusionMatrix = np.zeros((fileCount, fileCount))
        for fileIndex, file in enumerate(os.listdir(inputDir)):
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

    originalMatrix = computeConfusionMatrix(bestCaseResultsDirectory)
    remeshedMatrix = computeConfusionMatrix(remeshedResultsDirectory)

    globalMin = min(originalMatrix.min(), remeshedMatrix.min())
    globalMax = max(originalMatrix.max(), remeshedMatrix.max())

    print('Confusion matrices are shown. Close both windows to continue.')
    print()

    plt.title("AUGMENTED_Best")
    heatmap = plt.imshow(originalMatrix, cmap='hot', interpolation='nearest', vmin=globalMin, vmax=globalMax)

    plt.figure(2)
    plt.title("AUGMENTED_Rem")
    plt.imshow(remeshedMatrix, cmap='hot', interpolation='nearest', vmin=globalMin, vmax=globalMax)
    plt.colorbar(heatmap)  # , format=ticker.FuncFormatter(fmt))
    plt.show()

def runAllToAllObjectSearch():
    while True:
        run_menu = TerminalMenu([
            "Compute and compare search results for one non-remeshed query using the original QUICCI descriptor",
            "Compute and compare search results for one non-remeshed query using the modified QUICCI descriptor",
            "Compute and compare search results for one remeshed query using the original QUICCI descriptor",
            "Compute and compare search results for one remeshed query using the modified QUICCI descriptor",
            "Compute Table 1 and Figure 13 from results computed by authors",
            "Compute Table 1 and Figure 13 from scratch",
            "back"], title='-- Reproduce Table 1 and Figure 13: All to all retrieval --')

        choice = run_menu.show() + 1

        if choice <= 4:
            if choice <= 2:
                inputBasePath = 'output/augmented_dataset_original'
            else:
                inputBasePath = 'output/augmented_dataset_remeshed'

            randomMeshIndex = random.randint(0, len(os.listdir(inputBasePath)))
            queryMeshFileName = os.listdir(inputBasePath)[randomMeshIndex]
            queryMesh = os.path.join(inputBasePath, queryMeshFileName)

            print()
            print('Randomly selected query mesh:', queryMesh)
            print()

            if choice == 1:
                runSingleAllToAll(queryMesh, False, True)
            if choice == 2:
                runSingleAllToAll(queryMesh, False, False)
            if choice == 3:
                runSingleAllToAll(queryMesh, True, True)
            if choice == 4:
                runSingleAllToAll(queryMesh, True, False)
        if choice == 5:
            outputTable = PrettyTable(['QUICCI', 'AUGMENTED_Best', 'AUGMENTED_Rem'])
            outputTable.align = "l"
            outputTable.add_row(['Original', processAllToAllResultsDirectory(computeAllToAllReferenceDirectory(False, True)),
                                             processAllToAllResultsDirectory(computeAllToAllReferenceDirectory(True, True))])
            outputTable.add_row(['Modified', processAllToAllResultsDirectory(computeAllToAllReferenceDirectory(False, False)),
                                             processAllToAllResultsDirectory(computeAllToAllReferenceDirectory(True, False))])
            print()
            print('Table 1:')
            print()
            print(outputTable)
            print()
            computeHeatmaps(computeAllToAllReferenceDirectory(False, False),
                            computeAllToAllReferenceDirectory(True, False))
        if choice == 6:
            for inputBasePath in ['output/augmented_dataset_original', 'output/augmented_dataset_remeshed']:
                for file in os.listdir(inputBasePath):
                    runSingleAllToAll(os.path.join(inputBasePath, file), False, True)
                    runSingleAllToAll(os.path.join(inputBasePath, file), False, False)
                    runSingleAllToAll(os.path.join(inputBasePath, file), True, True)
                    runSingleAllToAll(os.path.join(inputBasePath, file), True, False)

            outputTable = PrettyTable(['QUICCI', 'AUGMENTED_Best', 'AUGMENTED_Rem'])
            outputTable.align = "l"
            outputTable.add_row(
                ['Original', processAllToAllResultsDirectory(computeAllToAllReplicatedDirectory(False, True)),
                             processAllToAllResultsDirectory(computeAllToAllReplicatedDirectory(True, True))])
            outputTable.add_row(
                ['Modified', processAllToAllResultsDirectory(computeAllToAllReplicatedDirectory(False, False)),
                             processAllToAllResultsDirectory(computeAllToAllReplicatedDirectory(True, False))])
            print()
            print('Table 1:')
            print()
            print(outputTable)
            print()
            computeHeatmaps(computeAllToAllReplicatedDirectory(False, False),
                            computeAllToAllReplicatedDirectory(True, False))
        if choice == 7:
            return



def computeShrec16Benchmark(partiality, resolution):
    outputFile = 'output/Figure_16_SHREC16_benchmark/results_' + resolution + 'x' + resolution \
                             + '_' + partiality + '_partiality.json'
    run_command_line_command('bin/build' + resolution + 'x' + resolution + '/objectSearch '
                             '--index-directory=output/dissimilarity_tree/index' + resolution + 'x' + resolution + ' '
                             '--haystack-directory=input/SHREC2016_partial_retrieval/complete_objects '
                             '--query-directory=input/SHREC2016_partial_retrieval/queries_artificial/Q' + partiality + ' '
                             '--resultsPerQueryImage=1 '
                             '--randomSeed=' + mainEvaluationRandomSeed + ' '
                             '--support-radius=' + shrec2016_support_radius + ' '
                             '--consensus-threshold=10 '
                             '--force-gpu=' + str(gpuID) + ' '
                             '--output-file=' + outputFile)

    # Load label files provided by benchmark authors
    with open('input/SHREC2016_partial_retrieval/Hampson_query_cla_6.txt', 'r') as inFile:
        queryLabelsFileContents = inFile.readlines()
    with open('input/SHREC2016_partial_retrieval/Hampson_target_cla_6.txt', 'r') as inFile:
        targetLabelsFileContents = inFile.readlines()

    # Filter data
    queryLabels = [x.split('\t')[0] for x in queryLabelsFileContents]
    targertLabels = [x.split(' ')[0] for x in targetLabelsFileContents]

    # Build index
    targetIndices = {}
    for index, label in enumerate(targertLabels):
        # The file is organised as an indexed list. Row 1 is object 1, row 2 is object 2, etc.
        # We can therefore translate object indices to dataset object filenames
        targetIndices[label] = 'T' + str(index + 1) + '.obj'

    # Use this to compute the 'answer key' for the retrieval task
    correctMatches = [targetIndices[x] for x in queryLabels]

    with open(outputFile, 'r') as inFile:
        computedMatchesFileContents = json.loads(inFile.read())

    computedMatches = len(queryLabels) * [None]
    for entry in computedMatchesFileContents['results']:
        queryIndex = int(os.path.basename(entry['queryFile'])[1:-4]) - 1
        bestMatchingObject = os.path.basename(entry['searchResults'][0]['objectFilePath'])
        computedMatches[queryIndex] = bestMatchingObject

    print()
    print('Results:')
    print()
    outputTable = PrettyTable(
        ['Computed nearest neighbour', 'Benchmark label', 'Do the left two columns match?'])
    outputTable.align = "l"
    for index, matchedObjectName in enumerate(correctMatches):
        outputTable.add_row([matchedObjectName, computedMatches[index],
                             'Yes' if matchedObjectName == computedMatches[index] else 'NO!!!!'])
    print(outputTable)
    print()
    print('For the results to match those shown in the paper, the left two columns of the table must match entirely.')
    print()

def computePipelineEvaluationAuthorReferenceFileName(queryMode, threshold, resolution):
    filename = 'results_' + ('bestcase' if queryMode == "Best Case" else 'remeshed') \
                          + '_threshold' + threshold \
                          + '_resolution' + resolution + '.json'
    return 'input/precomputed_results/figure14_and_15/' + filename

def computePipelineEvaluationOutputFileName(queryMode, threshold, resolution):
    filename = 'results_pipelineEvaluation_' + ('bestcase' if queryMode == "Best Case" else 'remeshed') \
                                             + '_resolution' + resolution \
                                             + '_threshold' + threshold + '.json'
    return 'output/Figure_14_and_15_Pipeline_Evaluation/computed_results/' + filename

def runQuerySet(randomBatchSize):
    global pipelineEvaluation_resolution
    global pipelineEvaluation_consensusThreshold
    global pipelineEvaluation_queryMode

    startIndex = random.randint(0, len(os.listdir('input/SHREC2016_partial_retrieval/complete_objects')) - randomBatchSize)
    endIndex = startIndex + randomBatchSize


    queryPath = 'output/augmented_dataset_original' if pipelineEvaluation_queryMode == 'Best Case' else 'output/augmented_dataset_remeshed'
    resolution = pipelineEvaluation_resolution
    consensusThreshold = pipelineEvaluation_consensusThreshold

    outputFile = computePipelineEvaluationOutputFileName(pipelineEvaluation_queryMode, consensusThreshold, resolution)

    run_command_line_command('bin/build' + resolution + '/objectSearch '
         '--index-directory=output/dissimilarity_tree/index' + resolution + ' '
         '--haystack-directory=input/SHREC2016_partial_retrieval/complete_objects '
         '--query-directory=' + queryPath + ' '
         '--resultsPerQueryImage=1 '
         '--randomSeed=' + mainEvaluationRandomSeed + ' '
         '--support-radius=' + shrec2016_support_radius + ' '
         '--consensus-threshold=' + consensusThreshold + ' '
         '--force-gpu=' + str(gpuID) + ' '
         '--output-file=' + outputFile + ' '
         '--subset-start-index=' + str(startIndex) + ' '
         '--subset-end-index=' + str(endIndex))

    with open(outputFile, 'r') as inFile:
        computedResults = json.loads(inFile.read())

    referenceFileName = computePipelineEvaluationAuthorReferenceFileName(pipelineEvaluation_queryMode, consensusThreshold, resolution)
    with open(referenceFileName, 'r') as inFile:
        referenceResults = json.loads(inFile.read())

    outputTable = PrettyTable(['Query Object', 'Execution Time (computed)', 'Best Match (computed)', '',
                               'Execution Time (authors)', 'Best Match (authors)'])
    outputTable.align = "l"
    for i in range(0, randomBatchSize):
        outputTable.add_row([os.path.basename(computedResults['results'][i]['queryFile']),
                             computedResults['results'][i]['executionTimeSeconds'],
                             os.path.basename(computedResults['results'][i]['searchResults'][0]['objectFilePath']),
                             '',
                             referenceResults['results'][startIndex + i]['executionTimeSeconds'],
                             os.path.basename(referenceResults['results'][startIndex + i]['searchResults'][0]['objectFilePath'])])
    print()
    print(outputTable)
    print()

def evaluatePipelineResults(inputFile):
    correctCount = 0

    maxTimeSeconds = 600
    histogramPrecision = 1
    timeHistogram = (histogramPrecision * maxTimeSeconds) * [0]
    processedTimeSlices = []

    with open(inputFile, 'r') as inFile:
        fileContents = json.loads(inFile.read())

        for queryIndex, result in enumerate(fileContents['results']):
            if result['searchResults'][0]['objectID'] == queryIndex:
                correctCount += 1
            executionTime = result['executionTimeSeconds']
            timeHistogramIndex = int(executionTime * histogramPrecision)
            timeHistogram[timeHistogramIndex] += 1

        for i in range(0, maxTimeSeconds * histogramPrecision):
            processedTimeSlices.append((float(i) / histogramPrecision, timeHistogram[i]))

    return correctCount, len(fileContents['results']), processedTimeSlices

def notifyMissingPipelineResults(queryMode, threshold, resolution):
    global pipelineEvaluation_resolution
    global pipelineEvaluation_consensusThreshold
    global pipelineEvaluation_queryMode
    print()
    print('It looks like you have missing prerequisite search results.')
    if ask_for_confirmation("Would you like me to apply the settings you need for the next missing set?"):
        pipelineEvaluation_queryMode = queryMode
        pipelineEvaluation_consensusThreshold = threshold
        pipelineEvaluation_resolution = resolution
        print()
        print('Done. Pick any of the top three run options to compute results.')
    print()

class MissingInputFileException(BaseException):
    pass


def runPipelineEvaluation():
    global pipelineEvaluation_resolution
    global pipelineEvaluation_consensusThreshold
    global pipelineEvaluation_queryMode

    os.makedirs('output/Figure_14_and_15_Pipeline_Evaluation/computed_results', exist_ok=True)

    while True:
        run_menu = TerminalMenu([
            "Run and compare results for 10 random objects",
            "Run and compare results for 50 random objects",
            "Run and compare results for all objects",
            "Configure descriptor resolution (currently: " + pipelineEvaluation_resolution + ')',
            "Configure vote threshold (currently: " + pipelineEvaluation_consensusThreshold + ')',
            "Configure query dataset (currently: " + pipelineEvaluation_queryMode + ')',
            "Compute Figure 14 based on results computed by Authors",
            "Compute Figure 14 based on replicated results",
            "Compute Figure 15 based on results computed by Authors",
            "Compute Figure 15 based on replicated results",
            "back"], title='-- Reproduce Figure 14 and 15: Pipeline Evaluation --')

        choice = run_menu.show() + 1

        if choice == 1:
            runQuerySet(10)
        if choice == 2:
            runQuerySet(50)
        if choice == 3:
            runQuerySet(len(os.listdir('input/SHREC2016_partial_retrieval/complete_objects')))
        if choice == 4:
            configure_resolution_menu = TerminalMenu([
                "Compute descriptors with resolution 32x32",
                "Compute descriptors with resolution 64x64",
                "Compute descriptors with resolution 96x96",
                "back"], title='-- Configure Descriptor Resolution --')
            configure_choice = configure_resolution_menu.show() + 1
            if configure_choice == 1:
                pipelineEvaluation_resolution = '32x32'
            if configure_choice == 2:
                pipelineEvaluation_resolution = '64x64'
            if configure_choice == 3:
                pipelineEvaluation_resolution = '96x96'
        if choice == 5:
            configure_resolution_menu = TerminalMenu([
                "Set vote threshold to 10",
                "Set vote threshold to 25",
                "Set vote threshold to 50",
                "back"], title='-- Configure Vote Threshold --')
            configure_choice = configure_resolution_menu.show() + 1
            if configure_choice == 1:
                pipelineEvaluation_consensusThreshold = '10'
            if configure_choice == 2:
                pipelineEvaluation_consensusThreshold = '25'
            if configure_choice == 3:
                pipelineEvaluation_consensusThreshold = '50'
        if choice == 6:
            configure_resolution_menu = TerminalMenu([
                "Use the Best Case dataset as query objects",
                "Use the Remeshed dataset as query objects",
                "back"], title='-- Configure Query Dataset --')
            configure_choice = configure_resolution_menu.show() + 1
            if configure_choice == 1:
                pipelineEvaluation_queryMode = 'Best Case'
            if configure_choice == 2:
                pipelineEvaluation_queryMode = 'Remeshed'
        if choice == 7:
            chart = [['', 'Threshold 10', 'Threshold 25', 'Threshold 50'], ['', 0, 0, 0], ['', 0, 0, 0], ['', 0, 0, 0], ['', 0, 0, 0], ['', 0, 0, 0], ['', 0, 0, 0]]
            for thresholdIndex, threshold in enumerate(['10', '25', '50']):
                for resolutionIndex, resolution in enumerate(['32x32', '64x64', '96x96']):
                    for queryModeIndex, queryMode in enumerate(['Best Case', 'Remeshed']):
                        correctCount, totalObjectCount, _ = evaluatePipelineResults(
                            computePipelineEvaluationAuthorReferenceFileName(queryMode, threshold, resolution))
                        correctFraction = float(correctCount) / float(totalObjectCount)
                        chart[queryModeIndex + 2 * resolutionIndex + 1][0] = queryMode + ' (' + resolution + ')'
                        chart[queryModeIndex + 2 * resolutionIndex + 1][thresholdIndex + 1] = str(correctFraction)

            figure14_outputFile = 'output/Figure_14_and_15_Pipeline_Evaluation/Figure_14_precision_authors.csv'
            with open(figure14_outputFile, 'w') as outFile:
                for row in chart:
                    outFile.write(','.join(row) + '\n')

            print()
            print('Complete.')
            print('The output file has been written to:')
            print('    ' + figure14_outputFile)
            print()

        if choice == 8:
            chart = [['', 'Threshold 10', 'Threshold 25', 'Threshold 50'], ['', 0, 0, 0], ['', 0, 0, 0], ['', 0, 0, 0],
                     ['', 0, 0, 0], ['', 0, 0, 0], ['', 0, 0, 0]]
            try:
                for thresholdIndex, threshold in enumerate(['10', '25', '50']):
                    for resolutionIndex, resolution in enumerate(['32x32', '64x64', '96x96']):
                        for queryModeIndex, queryMode in enumerate(['Best Case', 'Remeshed']):
                            outputFilename = computePipelineEvaluationOutputFileName(queryMode, threshold, resolution)
                            if not os.path.exists(outputFilename):
                                notifyMissingPipelineResults(queryMode, threshold, resolution)
                                raise MissingInputFileException()
                            correctCount, totalObjectCount, _ = evaluatePipelineResults(outputFilename)
                            correctFraction = float(correctCount) / float(totalObjectCount)
                            chart[queryModeIndex + 2 * resolutionIndex + 1][0] = queryMode + ' (' + resolution + ')'
                            chart[queryModeIndex + 2 * resolutionIndex + 1][thresholdIndex + 1] = str(correctFraction)
            except MissingInputFileException:
                continue
            figure14_outputFile = 'output/Figure_14_and_15_Pipeline_Evaluation/Figure_14_precision_replicated.csv'
            with open(figure14_outputFile, 'w') as outFile:
                for row in chart:
                    outFile.write(','.join(row) + '\n')

            print()
            print('Complete.')
            print('The output file has been written to:')
            print('    ' + figure14_outputFile)
            print()
        if choice == 9:
            _, _, figure15_bestCase = evaluatePipelineResults(computePipelineEvaluationAuthorReferenceFileName('Best Case', '10', '64x64').replace('results_', 'timings/results_'))
            _, _, figure15_remeshed = evaluatePipelineResults(computePipelineEvaluationAuthorReferenceFileName('Remeshed', '10', '64x64').replace('results_', 'timings/results_'))
            figure15_outputFile = 'output/Figure_14_and_15_Pipeline_Evaluation/Figure_15_queryTimes_authors.csv'

        if choice == 10:
            if not os.path.exists(computePipelineEvaluationOutputFileName('Best Case', '10', '64x64')):
                notifyMissingPipelineResults('Best Case', '10', '64x64')
                continue
            if not os.path.exists(computePipelineEvaluationOutputFileName('Remeshed', '10', '64x64')):
                notifyMissingPipelineResults('Remeshed', '10', '64x64')
                continue

            _, _, figure15_bestCase = evaluatePipelineResults(computePipelineEvaluationOutputFileName('Best Case', '10', '64x64'))
            _, _, figure15_remeshed = evaluatePipelineResults(computePipelineEvaluationOutputFileName('Remeshed', '10', '64x64'))
            figure15_outputFile = 'output/Figure_14_and_15_Pipeline_Evaluation/Figure_15_queryTimes_replicated.csv'

        if choice == 9 or choice == 10:
            with open(figure15_outputFile, 'w') as outFile:
                outFile.write('Time (s), Count (Best Case Queries), Count (Remeshed Queries)\n')
                for i in range(0, len(figure15_bestCase)):
                    outFile.write(str(figure15_bestCase[i][0]) + ', ' +
                                  str(figure15_bestCase[i][1]) + ', ' +
                                  str(figure15_remeshed[i][1]) + '\n')

            print()
            print('Complete.')
            print('The output file has been written to:')
            print('    ' + figure15_outputFile)
            print()

        if choice == 11:
            return

def runShrec16Queries():
    os.makedirs('output/Figure_16_SHREC16_benchmark', exist_ok=True)

    while True:
        run_menu = TerminalMenu([
            "Compute results for queries with 25% partiality with 32x32 bit descriptors",
            "Compute results for queries with 25% partiality with 64x64 bit descriptors",
            "Compute results for queries with 25% partiality with 96x96 bit descriptors",
            "Compute results for queries with 40% partiality with 32x32 bit descriptors",
            "Compute results for queries with 40% partiality with 64x64 bit descriptors",
            "Compute results for queries with 40% partiality with 96x96 bit descriptors",
            "back"], title='-- Reproduce Figure 16: SHREC\'16 benchmark --')

        choice = run_menu.show() + 1

        if choice == 1:
            computeShrec16Benchmark('25', '32')
        if choice == 2:
            computeShrec16Benchmark('25', '64')
        if choice == 3:
            computeShrec16Benchmark('25', '96')
        if choice == 4:
            computeShrec16Benchmark('40', '32')
        if choice == 5:
            computeShrec16Benchmark('40', '64')
        if choice == 6:
            computeShrec16Benchmark('40', '96')
        if choice == 7:
            return


def runMainMenu():
    while True:
        main_menu = TerminalMenu([
            "1. Install dependencies",
            "2. Download datasets",
            "3. Compile project",
            "4. Select which GPU to use (for multi GPU systems)",
            "5. Generate augmented SHREC'2016 dataset",
            "6. Compute descriptors",
            "7. Compute dissimilarity tree from descriptors",
            "8. Run vote counting experiment (Figure 3)",
            "9. Run average search result distance experiment (Figure 4)",
            "10. Compute average descriptor heatmap (Figure 6)",
            "11. Run dissimilarity tree execution time evaluation (Figure 10 and Figure 17)",
            "12. Run modified quicci evaluation (Figures 11 and 12)",
            "13. Run all to all object search (Table 1 and Figure 13)",
            "14. Run partial retrieval pipeline evaluation (Figures 14 and 15)",
            "15. Run SHREC'16 artificial benchmark (Figure 16)",
            "16. exit"], title='---------------------- Main Menu ----------------------')

        choice = main_menu.show() + 1

        if choice == 1:  # Done
            installDependenciesMenu()
        if choice == 2:  # TODO
            downloadDatasetsMenu()
        if choice == 3:  # Done
            compileProject()
        if choice == 4:  # Done
            configureGPU()
        if choice == 5:  # Done
            generateAugmentedDataset()
        if choice == 6:  # Done
            computeDescriptors()
        if choice == 7:  # Done
            computeDissimilarityTree()
        if choice == 8:  # Done
            runVoteCountProgressionExperiment()
        if choice == 9:  # Done
            computeAverageScoreChart()
        if choice == 10:  # Done
            computeBitsHeatmap()
        if choice == 11:  # Done
            runIndexEvaluation()
        if choice == 12:  # Done
            runModifiedQuicciEvaluation()
        if choice == 13:  # Done
            runAllToAllObjectSearch()
        if choice == 14:  # Done
            runPipelineEvaluation()
        if choice == 15:  # Done
            runShrec16Queries()
        if choice == 16:
            return

def runIntroSequence():
    print()
    print('Greetings!')
    print()
    print('This script is intended to reproduce various figures in an interactive')
    print('(and hopefully convenient) manner.')
    print()
    print('It is recommended you refer to the included PDF manual for instructions')
    print()
    runMainMenu()


if __name__ == "__main__":
    runIntroSequence()
