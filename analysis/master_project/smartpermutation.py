import matplotlib.pyplot as plt
import random


heatmap = []

with open('output/Figure_6_OccurrenceCountHeatmap/shrec16_occurrence_counts.txt') as lines:
    for line in lines:
        line = line.strip()
        heatmap.append(list(map(lambda x : int(x), line.split(", "))))

heatmap.reverse()

""" Should this be before or after reverse?
print("Heatmap counts:")
for r in heatmap:
    print(r)
""" 


flat_heatmap = sum(heatmap, [])

indexes = sorted(range(len(flat_heatmap)), key=lambda k: flat_heatmap[k])

print("Heatmap permutation:")
#for s in range(32):
#    print(indexes[s*32:s*32+32])
print(indexes)

cmap = plt.cm.get_cmap('hsv', max(indexes)*1.5)

for value in indexes: #[1:-1]:
    plt.plot(value % 32, value // 32, marker="s", color=cmap(indexes.index(value)))
plt.show()

# Verify that the heatmpat generated permutation is correct
# Incorporate as c++ permutation