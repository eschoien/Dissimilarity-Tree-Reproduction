import json
import matplotlib.pyplot as plt

paths = ['output/lsh/measurements/v4/partial_objects/permcount10/measurement-0.4-1000-10.json',
         'output/lsh/measurements/v_no_break/partial_objects/permcount10/measurement-0.4-1000-10.json',
         'output/lsh/measurements/v4/complete_objects/permcount10/measurement-0.4-1000-10.json',
         'output/lsh/measurements/v_no_break/complete_objects/permcount10/measurement-0.4-1000-10.json'
         ]

for path in paths:

    data = json.load(open(path))

    # Shows query objects (bottom) and the best matching object (top)
    plt.rcParams.update({'font.size': 24})
    plt.figure(figsize=(10, 10))

    queries = {}
    #trendingWrong = [[x, 0] for x in range(1, 384)]

    for d in data['results']:
        queries[d['queryFileID']] = d['bestMatches'][0]
        #trendingWrong[d['bestMatches'][0]-1][1] += 1

    #trendingWrong = sorted(trendingWrong, key=lambda x : x[1], reverse=True)
    #trendingWrong = list(map(lambda x: x[0], trendingWrong))

    #for i in range(1, 384):
    for queryID, bestMatchID in queries.items():
        #plt.plot(trendingWrong.index(queryID), trendingWrong.index(bestMatchID), marker='o', color='blue' if queryID == bestMatchID else 'red')
        plt.plot(bestMatchID, queryID, marker='o', color='blue' if queryID == bestMatchID else 'red')

    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Best match object ID")
    plt.ylabel("Query object ID")

    plt.tight_layout()
    plt.savefig(path[path.find('/v')+1:path.find('/perm')].replace('/', '_'), dpi=150)

    plt.show()


"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Partial, no-break
# data = json.load(open('output/lsh/measurements/v4/partial_objects/permcount10/measurement-0.4-1000-10.json'))
# Partial, with break
# data = json.load(open('output/lsh/measurements/v_no_break/partial_objects/permcount10/measurement-0.4-1000-10.json'))
# Complete, no break
# data = json.load(open('output/lsh/measurements/v4/complete_objects/permcount10/measurement-0.4-1000-10.json'))
# Complete, with break
data = json.load(open('output/lsh/measurements/v_no_break/complete_objects/permcount10/measurement-0.4-1000-10.json'))

# data = np.full((383, 383), 0, dtype=int)


table = [ [0]*383 for _ in range(383)]


for d in data['results']:
    table[int(d['queryFileID']-1)][int(d['bestMatches'][0]-1)] = 1


table = np.array(table)

  
# plotting a plot
# pixel_plot.add_axes()
  
# customizing plot
plt.title("pixel_plot")

pixel_plot = plt.imshow(table, cmap='Greys', interpolation='nearest', origin='lower')
  
# plt.colorbar(pixel_plot)
plt.gca().invert_yaxis()
  
# save a plot
# plt.savefig('pixel_plot.png')
  
# show plot
plt.show()
"""