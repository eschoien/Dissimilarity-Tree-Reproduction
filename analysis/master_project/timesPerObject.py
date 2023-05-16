import json
import pandas as pd
import matplotlib.pyplot as plt

#horizontal = json.load(open('horizontal/Dissimilarity-Tree-Reproduction/output/Figure_10_and_17_indexQueryTimes/computed_indexed_search_times.csv'))

#vertical = json.load(open('vertical/Dissimilarity-Tree-Reproduction/output/Figure_10_and_17_indexQueryTimes/computed_indexed_search_times.csv'))

minhash = json.load(open('output/lsh/measurements/v3/partial_objects/permcount10/measurement-0.4-500-10.json'))
disstree = json.load(open('output/dissTree/measurements/v4/partial_objects/measurement-383.json'))

def queryFileMeanDF(data):
    try:
        df = pd.DataFrame(data["results"])[['executionTimeSeconds', 'queryFile']]

        df['queryFile'] = df['queryFile'].apply(lambda x: x[x.find("/T")+2:-4].rjust(3, "0"))
    except:
        df = pd.DataFrame(data["results"])[['executionTimeSeconds', 'queryFileID']]

        df['queryFile'] = df['queryFileID'].apply(str)

    return df.groupby('queryFile').mean('executionTimeSeconds')

minhash_df = queryFileMeanDF(minhash)
disstree_df = queryFileMeanDF(disstree)

minhash_df.rename(columns = {'executionTimeSeconds':'mean_minhash'}, inplace = True)
disstree_df.rename(columns = {'executionTimeSeconds':'mean_disstree'}, inplace = True)

merged_df = pd.merge(minhash_df,disstree_df,on='queryFile').reset_index()

# print(merged_df)
# Change by= below to sort by different direction
merged_df.sort_values(by=['mean_disstree'], inplace=True)
# normalized_by = merged_df['mean_disstree']

# merged_df['minhash_normalized'] = merged_df['mean_minhash'] - normalized_by
# merged_df['disstree_normalized'] = merged_df['mean_disstree'] - normalized_by

plt.scatter(x=merged_df['queryFile'],y=merged_df['mean_minhash'], label="minhash", color="b")
plt.scatter(x=merged_df['queryFile'],y=merged_df['mean_disstree'], label="disstree", color="r")
plt.legend(loc="upper left")
plt.xticks([])
plt.show()