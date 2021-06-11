import os
import json

mode = 'original'

originalDir = '/mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/derived/original'
remeshedDir = '/mnt/NEXUS/Stash/SHREC2016/results/MECHANINJA/derived/remeshed'

haystackAreasFile = '/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/areas/haystack_remeshed.json'
originalAreasFile = '/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/areas/queries_original.json'
remeshedAreasFile = '/mnt/NEXUS/Stash/SHREC2016/results/TURBONINJA/areas/queries_remeshed.json'

with open(haystackAreasFile, 'r') as inputFile:
    haystackAreas = json.loads(inputFile.read())
with open(originalAreasFile, 'r') as inputFile:
    originalAreas = json.loads(inputFile.read())
with open(remeshedAreasFile, 'r') as inputFile:
    remeshedAreas = json.loads(inputFile.read())





if mode == 'remeshed':
    inputDir = remeshedDir
    queryAreas = remeshedAreas
else:
    inputDir = originalDir
    queryAreas = originalAreas





fileCount = 383
scoresByPartiality = []

for file in os.listdir(inputDir):
    print(file)
    with open(os.path.join(inputDir, file), 'r') as inputFile:
        fileContents = json.loads(inputFile.read())
    fileToFind = file.replace('.json', '.dat')
    for i in range(0, fileCount):
        if fileToFind == fileContents['results'][i]['name']:
            partiality = queryAreas['areas'][file.replace('.json', '.obj')] \
                       / haystackAreas['areas'][file.replace('.json', '.obj')]
            scoresByPartiality.append((partiality, i, fileContents['results'][i]['score']))
            break
print()
print('Number of times the correct file appeared at search result n:')

for i in range(0, fileCount):
    print(', '.join([str(x) for x in scoresByPartiality[i]]))