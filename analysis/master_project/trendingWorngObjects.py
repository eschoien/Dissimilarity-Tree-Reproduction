import json
import pandas as pd

pd.set_option('display.max_columns', None)  

data = json.load(open('analysis/preparatory_project_results/query_time_results/horizontal_measurements_indexed.json'))

wrong_input = {}
wrong_output = {}

for d in data['results']:
    queryFile = d["queryFileID"]
    bestMatchFile = d["bestSearchResultFileID"]
    if queryFile != bestMatchFile:

        if queryFile in wrong_input.keys():
            wrong_input[queryFile] += 1
        else:
            wrong_input[queryFile] = 1

        if bestMatchFile in wrong_output.keys():
            wrong_output[bestMatchFile] += 1
        else:
            wrong_output[bestMatchFile] = 1



#wrong_input_sorted = sorted(wrong_input.items(), key=lambda x:x[1])
#wrong_output_sorted = sorted(wrong_output.items(), key=lambda x:x[1])

df = pd.DataFrame(data={'ObjectID': range(1,384), 'wrongWhenInput': wrong_input.values(), 'wrongWhenOutput': wrong_output.values()})

print(df.to_string())