import json
import matplotlib.pyplot as plt

data = json.load(open('../../output/lsh/measurements-0.6-500.json'))

# Show relationship betweeen partial query objects (bottom)
# and best match objects (top) for queries
for d in data['results']:
    x1, y1 = [d["queryFileID"], d["bestMatchID"]], [0, 10]
    plt.plot(x1, y1, marker = 'o')

# plt.legend(loc="upper left")
# plt.xticks([])
plt.show()
