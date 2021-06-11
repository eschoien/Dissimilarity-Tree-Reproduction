import os
import json
import subprocess

machine = 'MECHANINJA'
mode = 'SHREC2016_derived'
skipExisting = True

if machine == 'TURBONINJA':
    queryMainFilesDirectory = '/mnt/WAREHOUSE/Datasets/SHREC2016-Partial-Shape-Queries/SHREC_Queries'
    executablePath = '/home/bart/git/Project-Symmetry/cmake-build-release-g-8/simplesearch'
    imageDirectory = '/mnt/NEXUS/Stash/SHREC2016/haystack'
    resultsDirectory = '/mnt/NEXUS/Stash/SHREC2016/results'

elif machine == 'MECHANINJA':
    queryMainFilesDirectory = '/home/bart/Datasets/SHREC2016-Partial-Shape-Queries/SHREC_Queries'
    executablePath = '/mnt/LEGACY/home/bart/git/Project-Symmetry/cmake-build-release/simplesearch'
    imageDirectory = '/mnt/WAREHOUSE2/Stash/SHREC2016/haystack'
    resultsDirectory = '/mnt/WAREHOUSE2/Stash/SHREC2016/results'

    derivedOriginalQueries = '/mnt/WAREHOUSE2/Stash/SHREC2016_generated_queries_original/'
    derivedRemeshedQueries = '/mnt/WAREHOUSE2/Stash/SHREC2016_generated_queries_remeshed/'

def searchImages(queryDirectory, outputDirectory):
    os.makedirs(outputDirectory, exist_ok=True)
    queriesToProcess = [f for f in os.listdir(queryDirectory) if os.path.isfile(os.path.join(queryDirectory, f))]
    print('Processing directory', queryDirectory, '->', outputDirectory)
    print('Found', len(queriesToProcess), 'files')
    for index, fileToProcess in enumerate(queriesToProcess):
        inputFilePath = os.path.join(queryDirectory, fileToProcess)
        dumpFilePath = os.path.join(outputDirectory, fileToProcess[0:-4] + '.json')
        print('\tProcessing file', (index + 1), '/', len(queriesToProcess), ':', fileToProcess)
        if os.path.exists(dumpFilePath) and skipExisting:
            print('\t\tOutput file already exists, skipping')
            continue
        subprocess.run(executablePath
                       + ' --query-mesh="' + inputFilePath + '"'
                       + ' --haystack-directory="' + imageDirectory + '"'
                       + ' --output-file="' + dumpFilePath + '"', shell=True)

print('Processing query objects..')
if mode == "SHREC2016":
    searchImages(os.path.join(queryMainFilesDirectory, 'Artificial', 'Q25'), os.path.join(resultsDirectory, 'Artificial', 'Q25'))
    searchImages(os.path.join(queryMainFilesDirectory, 'Artificial', 'Q40'), os.path.join(resultsDirectory, 'Artificial', 'Q40'))

    searchImages(os.path.join(queryMainFilesDirectory, 'Breuckmann', 'View1'), os.path.join(resultsDirectory, 'Breuckmann', 'View1'))
    searchImages(os.path.join(queryMainFilesDirectory, 'Breuckmann', 'View2'), os.path.join(resultsDirectory, 'Breuckmann', 'View2'))
    searchImages(os.path.join(queryMainFilesDirectory, 'Breuckmann', 'View3'), os.path.join(resultsDirectory, 'Breuckmann', 'View3'))

    searchImages(os.path.join(queryMainFilesDirectory, 'Kinect', 'View1'), os.path.join(resultsDirectory, 'Kinect', 'View1'))
    searchImages(os.path.join(queryMainFilesDirectory, 'Kinect', 'View2'), os.path.join(resultsDirectory, 'Kinect', 'View2'))
    searchImages(os.path.join(queryMainFilesDirectory, 'Kinect', 'View3'), os.path.join(resultsDirectory, 'Kinect', 'View3'))
elif mode == "SHREC2016_derived":
    searchImages(derivedOriginalQueries, os.path.join(resultsDirectory, 'derived', 'original_whamming'))
    searchImages(derivedRemeshedQueries, os.path.join(resultsDirectory, 'derived', 'remeshed_whamming'))
