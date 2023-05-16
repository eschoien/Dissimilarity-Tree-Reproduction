import json
import matplotlib.pyplot as plt

# Partial, no-break
# data = json.load(open('output/lsh/measurements/v4/partial_objects/permcount10/measurement-0.4-1000-10.json'))
# Partial, with break
# data = json.load(open('output/lsh/measurements/v_no_break/partial_objects/permcount10/measurement-0.4-1000-10.json'))
# Complete, no break
# data = json.load(open('output/lsh/measurements/v4/complete_objects/permcount10/measurement-0.4-1000-10.json'))
# Complete, with break
data = json.load(open('output/lsh/measurements/v_no_break/complete_objects/permcount10/measurement-0.4-1000-10.json'))

# Shows query objects (bottom) and the best matching object (top)
for d in data['results']:
    x1, y1 = [d["queryFileID"], d["bestMatches"][0]], [0, 10]
    color = "blue" if x1[0] == x1[1] else "red"
    plt.plot(x1, y1, marker = 'o', color=color)
    
plt.show()