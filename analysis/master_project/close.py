import json
import os
import matplotlib.pyplot as plt
import numpy as np

data = json.load(open('output/lsh/measurements/permcount10/measurement-0.6-200-10.json'))

x = []
y1 = []
y2 = []

for d in data['results']:
    guessed = d['bestMatchScore']
    correct = d['queryFileScore']

    qid = d['queryFileID']
    x.append(qid)

    # close = (correct / guessed) * 100
    # y1.append(close)
    # y2.append(100-close)

    y1.append(guessed)
    y2.append(correct-guessed)

    # print(f'{qid}: {round(close, 2)}%')

plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.show()