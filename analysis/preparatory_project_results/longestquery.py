import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

def max_query(data):
    queryfile = ''
    maxnum = 0
    for i in data['results']:
        if int(i['executionTimeSeconds']) > maxnum:
            maxnum = int(i['executionTimeSeconds'])
            queryfile = i['queryFile']
    
    return queryfile

def sortbytime(data):
    sorted_data = sorted(data['results'], key=lambda x : x['executionTimeSeconds'], reverse=True)
    result = []
    other_result = {}
    amount = {}
    for i in sorted_data:
        queryfile = i['queryFile'][34:-4]
        sec = i['executionTimeSeconds']
        try:
            other_result[queryfile] += sec
            amount[queryfile] += 1
        except:
            other_result[queryfile] = sec
            amount[queryfile] = 1
        
    for k, v in amount.items():
        other_result[k] = other_result[k] / v

    sorted_result = {k: v for k, v in sorted(other_result.items(), key=lambda item: int(item[0][1:]))}
    
    return sorted_result

    
    
f = open('query_time_results/combined_measurements_indexed.json')
combined_data = json.load(f)

f = open('query_time_results/horizontal_measurements_indexed.json')
horizontal_data = json.load(f)

f = open('query_time_results/vertical_measurements_indexed.json')
vertical_data = json.load(f)

f = open('query_time_results/figure10_indexed_search_100000.json')
author_data = json.load(f)

# print(max_query(combined_data))
sorted_horizontal = sortbytime(horizontal_data)
sorted_vertical = sortbytime(vertical_data)
sorted_combinded = sortbytime(combined_data)

# index = 0
# for k, v in sorted_horizontal.items():
#     print(f'{k}: Horizontal: {index}, Vertical: {list(sorted_vertical.keys()).index(k)}, Combined: {list(sorted_combinded.keys()).index(k)}')

#     index += 1

objects = []
h_time = []
v_time = []
c_time = []

# index = 0
# for k, v in sorted_horizontal.items():
#     objects.append(k)
#     h_time.append(v)
#     index+=1
#     if index == 10:
#         break

# index = 0
# for k, v in sorted_vertical.items():
#     v_time.append(v)
#     index+=1
#     if index == 10:
#         break

# index = 0
# for k, v in sorted_combinded.items():
#     c_time.append(v)
#     index+=1
#     if index == 10:
#         break

for i in range(10):
    if i == 4:
        obj = f'T{380}'
        objects.append(obj)
        h_time.append(sorted_horizontal[obj])
        v_time.append(sorted_vertical[obj])
        c_time.append(sorted_combinded[obj])
        continue
    
    if i == 7:
        obj = f'T{275}'
        objects.append(obj)
        h_time.append(sorted_horizontal[obj])
        v_time.append(sorted_vertical[obj])
        c_time.append(sorted_combinded[obj])
        continue

    if i == 5:
        obj = f'T{373}'
        objects.append(obj)
        h_time.append(sorted_horizontal[obj])
        v_time.append(sorted_vertical[obj])
        c_time.append(sorted_combinded[obj])
        continue
        
    x = random.randint(1, 383)
    obj = f'T{x}'
    objects.append(obj)
    h_time.append(sorted_horizontal[obj])
    v_time.append(sorted_vertical[obj])
    c_time.append(sorted_combinded[obj])

data = {
    'Object': objects,
    'Horizontal_time': h_time,
    'vertical_time': v_time,
    'combined_time': c_time
}

df = pd.DataFrame(data)

N = len(objects)
ind = np.arange(N) 
width = 0.25

bar1 = plt.bar(ind, df['Horizontal_time'], width, color='b')
bar2 = plt.bar(ind+width, df['vertical_time'], width, color='r')
bar3 = plt.bar(ind+width*2, df['combined_time'], width, color='y')

plt.xlabel("Objects")
plt.ylabel("Seconds")
# plt.title("test")

plt.xticks(ind+width, objects)
plt.legend((bar1, bar2, bar3), ('Horizontal', 'Vertical', 'Combined'))
plt.show()