import matplotlib.pyplot as plt
import os
import json
import numpy as np

directory = 'output/lsh/measurements/permcount10'
for filename in os.listdir(directory):
    data = json.load(open(directory+'/'+str(filename)))

    for r in data["results"]:
        plt.plot([r["executionTimeSeconds"]], data["descriptorsPerObjectLimit"], marker = 'o')

plt.show()