import json
import matplotlib.pyplot as plt

data = json.load(open('../preparatory_project_results/query_time_results/horizontal_measurements_indexed.json'))

# correct = 0
# files = [0] * 383


# Show relationship betweeen partial query objects (bottom)
# and best match objects (top) for queries (old project)

# other figures in comments
for d in data['results']:
    #if d["queryFileID"] == d["bestSearchResultFileID"]:
    #   correct += 1
    #   continue

    x1, y1 = [d["queryFileID"], d["bestSearchResultFileID"]], [0, 10]
    plt.plot(x1, y1, marker = 'o')
    
    #files[d["bestSearchResultFileID"]] += 1
    #files[d["queryFileID"]] -= 1

# plt.scatter(x=range(1,384),y=files, label="Horizontal", color="b")

# print(correct/10000, correct)
# print(sum(files))
plt.show()