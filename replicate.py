import json
import os
import shutil
import subprocess
import random
import sys
import multiprocessing
import hashlib

from scripts.simple_term_menu import TerminalMenu

from prettytable import PrettyTable

gpuID = 0
mainEvaluationRandomSeed = '725948161'
shrec2016_support_radius = '100'
descriptorWidthBits = 64
indexGenerationMode = 'GPU'

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

def downloadDatasetsMenu():
    download_menu = TerminalMenu([
        "Download all",
        "Download SHREC 2016 partial 3D shape dataset (7.9GB download, ~52.5GB extracted, needed for all Figures)",
        "Download experiment results generated by authors (~1.6GB download, 13.8GB extracted, needed for Figures 8-11)",
        "back"], title='------------------ Download Datasets ------------------')

    while True:
        choice = download_menu.show()
        os.makedirs('input/download/', exist_ok=True)

        if choice == 0 or choice == 1:
            if not os.path.isfile('input/download/SHREC17.7z') or ask_for_confirmation('It appears the SHREC 2017 dataset has already been downloaded. Would you like to download it again?'):
                print('Downloading SHREC 2017 dataset..')
                run_command_line_command('wget --output-document SHREC17.7z https://data.mendeley.com/public-files/datasets/ysh8p862v2/files/607f79cd-74c9-4bfc-9bf1-6d75527ae516/file_downloaded', 'input/download/')
            print()
            os.makedirs('input/SHREC17', exist_ok=True)
            run_command_line_command('p7zip -k -d download/SHREC17.7z', 'input/')
            print('Download and extraction complete. You may now delete the file input/download/SHREC17.7z if you need the disk space.')
            print()

        if choice == 0 or choice == 2:
            if not os.path.isfile('input/download/results_computed_by_authors.7z') or ask_for_confirmation('It appears the results archive file has already been downloaded. Would you like to download it again?'):
                print('Downloading results archive file..')
                run_command_line_command('wget --output-document results_computed_by_authors.7z https://data.mendeley.com/public-files/datasets/p7g8fz82rk/files/29a722cc-b7b5-456a-a096-5d8ac55d6881/file_downloaded', 'input/download/')
            print()
            run_command_line_command('p7zip -k -d download/results_computed_by_authors.7z', 'input/')

            print()
            if not os.path.isfile('input/download/results_computed_by_authors_quicci_fpfh.7z') or ask_for_confirmation('It appears the second results archive file has already been downloaded. Would you like to download it again?'):
                print('Downloading results archive file..')
                run_command_line_command('wget --output-document results_computed_by_authors_quicci_fpfh.7z https://data.mendeley.com/public-files/datasets/k9j5ymry29/files/519b9cab-71a7-40fa-924e-10cf9b7905d7/file_downloaded', 'input/download/')
            print()
            run_command_line_command('p7zip -k -d download/results_computed_by_authors_quicci_fpfh.7z', 'input/')

            print()
            if not os.path.isfile('input/download/clutter_estimated_by_authors.7z') or ask_for_confirmation('It appears the clutter estimates file has already been downloaded. Would you like to download it again?'):
                print('Downloading clutter estimates file..')
                run_command_line_command('wget --output-document clutter_estimated_by_authors.7z https://data.mendeley.com/public-files/datasets/p7g8fz82rk/files/37d353c5-7fd4-4488-a94a-97bb58dc722d/file_downloaded', 'input/download/')
            print()
            run_command_line_command('p7zip -k -d download/clutter_estimated_by_authors.7z', 'input/')

            print()
            print('Download and extraction complete. You may now delete the following files if you need the disk space:')
            print('- input/download/results_computed_by_authors.7z')
            print('- input/download/results_computed_by_authors_quicci_fpfh.7z')
            print('- input/download/clutter_estimated_by_authors.7z')
            print()

        if choice == 0 or choice == 3:
            if not os.path.isfile('input/download/distances_computed_by_authors.7z') or ask_for_confirmation('It appears the computed distances archive file has already been downloaded. Would you like to download it again?'):
                print('Downloading distance function distances computed by authors..')
                run_command_line_command('wget --output-document distances_computed_by_authors.7z https://data.mendeley.com/public-files/datasets/k9j5ymry29/files/b3fe4f65-bf36-4fa7-9d26-217a59e35e54/file_downloaded', 'input/download/')
            print()
            os.makedirs('input/SHREC17', exist_ok=True)
            run_command_line_command('p7zip -k -d download/distances_computed_by_authors.7z', 'input/')
            print('Download and extraction complete. You may now delete the file input/download/distances_computed_by_authors.7z if you need the disk space.')
            print()

        if choice == 4:
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

def configureGPU():
    global gpuID
    run_command_line_command('bin/build32x32/descriptorDumper --list-gpus')
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
    subprocess.run('bin/build' + descriptorWidth + 'x' + descriptorWidth + '/descriptorDumper'
                   + ' --input-file="' + inputFile
                   + '" --output-file="' + outputFile
                   + '" --support-radius=' + str(shrec2016_support_radius), shell=True)

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
            "back"], title='-- Compute descriptors --')
        choice = run_menu.show()
        if choice == 0:
            for index, descriptorwidth in enumerate(['32', '64', '96']):
                changeDescriptorWidth(int(descriptorwidth))
                print('Processing batch 1/5 in resolution ' + str(index + 1) + '/3 (' + descriptorwidth + 'x' + descriptorwidth + ')')
                computeDescriptorsFromDirectory('input/SHREC2016_partial_retrieval/complete_objects',
                                                'output/descriptors/complete_objects_' + descriptorwidth + 'x' + descriptorwidth,
                                                descriptorwidth)
                print('Processing batch 2/5 in resolution ' + str(index + 1) + '/3 (' + descriptorwidth + 'x' + descriptorwidth + ')')
                computeDescriptorsFromDirectory('output/augmented_dataset_original',
                                                'output/descriptors/augmented_dataset_original_' + descriptorwidth + 'x' + descriptorwidth,
                                                descriptorwidth)
                print('Processing batch 3/5 in resolution ' + str(index + 1) + '/3 (' + descriptorwidth + 'x' + descriptorwidth + ')')
                computeDescriptorsFromDirectory('output/augmented_dataset_remeshed',
                                                'output/descriptors/augmented_dataset_remeshed_' + descriptorwidth + 'x' + descriptorwidth,
                                                descriptorwidth)
                print('Processing batch 4/5 in resolution ' + str(index + 1) + '/3 (' + descriptorwidth + 'x' + descriptorwidth + ')')
                computeDescriptorsFromDirectory('input/SHREC2016_partial_retrieval/queries_artificial/Q25',
                                                'output/descriptors/shrec2016_25partiality_' + descriptorwidth + 'x' + descriptorwidth,
                                                descriptorwidth)
                print('Processing batch 5/5 in resolution ' + str(index + 1) + '/3 (' + descriptorwidth + 'x' + descriptorwidth + ')')
                computeDescriptorsFromDirectory('input/SHREC2016_partial_retrieval/queries_artificial/Q40',
                                                'output/descriptors/shrec2016_40partiality_' + descriptorwidth + 'x' + descriptorwidth,
                                                descriptorwidth)
        if choice == 1:
            print('Copying precomputed descriptors..')
            shutil.copytree('input/descriptors', 'output/descriptors', dirs_exist_ok=True)
        if choice == 2:
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
            "Configure GPU to use for index generation (current: GPU " + str(gpuID) + ')',
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
            configureGPU()
        if choice == 3:
            run_command_line_command('bin/build32x32/' + indexGenerationCommand
                                     + ' --index-directory=output/dissimilarity_tree/index32x32'
                                       ' --quicci-dump-directory=output/descriptors/complete_objects_32x32')
        if choice == 4:
            run_command_line_command('bin/build64x64/' + indexGenerationCommand
                                     + ' --index-directory=output/dissimilarity_tree/index64x64'
                                       ' --quicci-dump-directory=output/descriptors/complete_objects_64x64')
        if choice == 5:
            run_command_line_command('bin/build96x96/' + indexGenerationCommand
                                     + ' --index-directory=output/dissimilarity_tree/index96x96'
                                       ' --quicci-dump-directory=output/descriptors/complete_objects_96x96')
        if choice == 6:
            print('Copying precomputed index of 32x32 images..')
            shutil.copy('input/dissimilarity_tree_32x32/index.dat', 'output/dissimilarity_tree/index32x32/index.dat')
        if choice == 7:
            print('Copying precomputed index of 64x64 images..')
            shutil.copy('input/dissimilarity_tree_64x64/index.dat', 'output/dissimilarity_tree/index64x64/index.dat')
        if choice == 8:
            print('Copying precomputed index of 96x96 images..')
            shutil.copy('input/dissimilarity_tree_96x96/index.dat', 'output/dissimilarity_tree/index96x96/index.dat')
        if choice == 9:
            return

def runVoteCountProgressionExperiment():
    os.makedirs('output/Figure_3_voteCountProgression/input', exist_ok=True)
    shutil.copy('input/augmented_dataset_remeshed/T103.obj',
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
            "Configure GPU to use for index generation (current: GPU " + str(gpuID) + ')',
            "back"], title='-- Reproduce Figure 4: average relative distance chart --')

        choice = run_menu.show() + 1
        resultsFileToProcess = ''
        runCommand = 'bin/build64x64/indexedSearchBenchmark ' \
                     '--index-directory=output/dissimilarity_tree/index64x64 ' \
                     '--query-directory=input/augmented_dataset_original ' \
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
            with open('input/misc_precomputed_results/figure4_results_authors.json', 'r') as inFile:
                referenceResults = json.loads(inFile.read())

            outputTable = PrettyTable(['Score (computed)', 'File ID (computed)', 'Image ID (computed)', 'Score (authors)', 'File ID (authors)', 'Image ID (authors)'])
            for i in range(0, 50):
                outputTable.add_row([computedResults['results'][0]['searchResultFileIDs'][i]['score'],
                                     computedResults['results'][0]['searchResultFileIDs'][i]['fileID'],
                                     computedResults['results'][0]['searchResultFileIDs'][i]['imageID'],
                                     referenceResults['results'][objectIndexToTest]['searchResultFileIDs'][i]['score'],
                                     referenceResults['results'][objectIndexToTest]['searchResultFileIDs'][i]['fileID'],
                                    referenceResults['results'][objectIndexToTest]['searchResultFileIDs'][i]['imageID']])
            print(outputTable)

        if choice == 2:
            resultsFileToProcess = 'input/misc_precomputed_results/figure4_results_authors.json'
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
            configureGPU()
        if choice == 5:
            return

def computeBitsHeatmap():
    os.makedirs('output/Figure_6_OccurrenceCountHeatmap', exist_ok=True)
    run_command_line_command('bin/build64x64/occurrenceCounter '
                             '--index-directory=output/dissimilarity_tree/index64x64 '
                             '--output-file=output/Figure_6_OccurrenceCountHeatmap/shrec16_occurrence_counts.txt')
    run_command_line_command('python3 src/partialRetrieval/tools/shrec2016-runner/heatmap.py '
                             'output/Figure_6_OccurrenceCountHeatmap/shrec16_occurrence_counts.txt')


def runModifiedQuicciEvaluation():
    os.makedirs('output/Figure_11_and_12_unwantedBitEvaluation', exist_ok=True)
    run_command_line_command('bin/build64x64/edgeRemovalExperiment '
                             '--query-directory=input/augmented_dataset_original '
                             '--reference-object-directory=input/SHREC2016_partial_retrieval/complete_objects '
                             '--output-file=output/Figure_11_and_12_unwantedBitEvaluation/output.json '
                             '--force-gpu=' + str(gpuID) + ' '
                             '--support-radius=' + shrec2016_support_radius)

main_menu = TerminalMenu([
    "1. Install dependencies",
    "2. Download datasets",
    "3. Compile project",
    "4. Generate augmented SHREC'2016 dataset",
    "5. Compute descriptors",
    "6. Compute dissimilarity tree from descriptors",
    "7. Run vote counting experiment (Figure 3)",
    "8. Run average search result distance experiment (Figure 4)",
    "9. Compute average descriptor heatmap (Figure 6)",
    "10. Run dissimilarity tree evaluation (Figure 10)",
    "11. Run modified quicci evaluation (Figures 11 and 12)",
    "12. Run all to all object search (Table 1 and Figure 13)",
    "13. Run partial retrieval pipeline evaluation (Figures 14 and 15)",
    "14. Run SHREC'16 artificial benchmark (Figure 16)",
    "15. Run query duration evaluation (Figure 17)",
    "16. exit"], title='---------------------- Main Menu ----------------------')

def runMainMenu():
    while True:
        choice = main_menu.show() + 1

        if choice == 1:
            installDependenciesMenu()
        if choice == 2:
            downloadDatasetsMenu()
        if choice == 3:
            compileProject()
        if choice == 4:
            generateAugmentedDataset()
        if choice == 5:
            computeDescriptors()
        if choice == 6:
            computeDissimilarityTree()
        if choice == 7:
            runVoteCountProgressionExperiment()
        if choice == 8:
            computeAverageScoreChart()
        if choice == 9:
            computeBitsHeatmap()
        if choice == 10:
            pass
        if choice == 11:
            runModifiedQuicciEvaluation()
        if choice == 12:
            pass
        if choice == 13:
            pass
        if choice == 14:
            pass
        if choice == 15:
            pass
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