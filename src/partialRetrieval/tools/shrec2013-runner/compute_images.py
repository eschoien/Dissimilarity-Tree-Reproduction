import os
import subprocess
import sys

#machine='TURBONINJA'
machine = 'MECHANINJA'

if machine == 'TURBONINJA':
	querySampleFilesDirectory = '/mnt/WAREHOUSE/Datasets/SHREC2016-Partial-Shape-Queries/SHREC_Examples/Query_Examples'
	queryMainFilesDirectory = '/mnt/WAREHOUSE/Datasets/SHREC2016-Partial-Shape-Queries/SHREC_Queries'
	haystackFilesDirectory = '/mnt/WAREHOUSE/Datasets/SHREC2016-Partial-Shape-Queries/SHREC_Examples/Target'
	executablePath = '/home/bart/git/Project-Symmetry/cmake-build-release-g-8/descriptorDumper'
	dumpDirectory = '/mnt/NEXUS/Stash/SHREC2016'

elif machine == 'MECHANINJA':
	querySampleFilesDirectory = '/home/bart/Datasets/SHREC2016-Partial-Shape-Queries/SHREC_Examples/Query_Examples'
	queryMainFilesDirectory = '/home/bart/Datasets/SHREC2016-Partial-Shape-Queries/SHREC_Queries'
	haystackFilesDirectory = '/home/bart/Datasets/SHREC2016-Partial-Shape-Queries/SHREC_Examples/Target'
	executablePath = '/mnt/LEGACY/home/bart/git/Project-Symmetry/cmake-build-release/descriptorDumper'
	dumpDirectory = '/mnt/WAREHOUSE2/Stash/SHREC2016'


	haystackFilesDirectory = '/home/bart/Datasets/SHREC2013-Large-Scale-Partial-Retrieval/shrec2013/Target'
	dumpDirectory = '/mnt/WAREHOUSE2/Stash/SHREC2013/descriptors'

querySampleDumpDirectory = os.path.join(dumpDirectory, 'query', 'sample')
queryMainDumpDirectory = os.path.join(dumpDirectory, 'query', 'main')
haystackDumpDirectory = os.path.join(dumpDirectory, 'haystack')

def computeImages(inputDirectory, outputDirectory):
    os.makedirs(outputDirectory, exist_ok=True)
    filesToProcess = [f for f in os.listdir(inputDirectory) if os.path.isfile(os.path.join(inputDirectory, f))]
    print('Processing directory', inputDirectory, '->', outputDirectory)
    print('Found', len(filesToProcess), 'files')
    for index, fileToProcess in enumerate(filesToProcess):
        inputFilePath = os.path.join(inputDirectory, fileToProcess)
        dumpFilePath = os.path.join(outputDirectory, fileToProcess[0:-4] + '.dat')
        print('\tProcessing file', (index + 1), '/', len(filesToProcess), ':', fileToProcess)

        subprocess.run(executablePath
                       + ' --input-file="' + inputFilePath
                       + '" --output-file="' + dumpFilePath
                       + '" --support-radius=0.5', shell=True)

print('Processing query objects..')
#computeImages(os.path.join(queryMainFilesDirectory, 'Artificial', 'Q25'), os.path.join(dumpDirectory, 'Artificial', 'Q25'))
#computeImages(os.path.join(queryMainFilesDirectory, 'Artificial', 'Q40'), os.path.join(dumpDirectory, 'Artificial', 'Q40'))

#computeImages(os.path.join(queryMainFilesDirectory, 'Breuckmann', 'View1'), os.path.join(dumpDirectory, 'Breuckmann', 'View1'))
#computeImages(os.path.join(queryMainFilesDirectory, 'Breuckmann', 'View2'), os.path.join(dumpDirectory, 'Breuckmann', 'View2'))
#computeImages(os.path.join(queryMainFilesDirectory, 'Breuckmann', 'View3'), os.path.join(dumpDirectory, 'Breuckmann', 'View3'))

#computeImages(os.path.join(queryMainFilesDirectory, 'Kinect', 'View1'), os.path.join(dumpDirectory, 'Kinect', 'View1'))
#computeImages(os.path.join(queryMainFilesDirectory, 'Kinect', 'View2'), os.path.join(dumpDirectory, 'Kinect', 'View2'))
#computeImages(os.path.join(queryMainFilesDirectory, 'Kinect', 'View3'), os.path.join(dumpDirectory, 'Kinect', 'View3'))

#computeImages(querySampleFilesDirectory, querySampleDumpDirectory)


print()
print('Processing haystack objects..')
computeImages(haystackFilesDirectory, haystackDumpDirectory)

