import json
import pandas as pd
import matplotlib.pyplot as plt

#horizontal = json.load(open('horizontal/Dissimilarity-Tree-Reproduction/output/Figure_10_and_17_indexQueryTimes/computed_indexed_search_times.csv'))

#vertical = json.load(open('vertical/Dissimilarity-Tree-Reproduction/output/Figure_10_and_17_indexQueryTimes/computed_indexed_search_times.csv'))

combined = json.load(open('query_time_results/combined_measurements_indexed.json'))
horizontal = json.load(open('query_time_results/horizontal_measurements_indexed.json'))
vertical = json.load(open('query_time_results/vertical_measurements_indexed.json'))

def queryFileMeanDF(data):

    df = pd.DataFrame(data["results"])[['executionTimeSeconds', 'queryFile']]

    df['queryFile'] = df['queryFile'].apply(lambda x: x[x.find("/T")+2:-4].rjust(3, "0"))

    return df.groupby('queryFile').mean('executionTimeSeconds')

horizontal_df = queryFileMeanDF(horizontal)
vertical_df = queryFileMeanDF(vertical)
combined_df = queryFileMeanDF(combined)

horizontal_df.rename(columns = {'executionTimeSeconds':'mean_horizontal'}, inplace = True)
vertical_df.rename(columns = {'executionTimeSeconds':'mean_vertical'}, inplace = True)
combined_df.rename(columns = {'executionTimeSeconds':'mean_combined'}, inplace = True)

merged_df = pd.merge(pd.merge(horizontal_df,vertical_df,on='queryFile'),combined_df,on='queryFile').reset_index()

# print(merged_df)
# Change by= below to sort by different direction
merged_df.sort_values(by=['mean_combined'], inplace=True)

plt.scatter(x=merged_df['queryFile'],y=merged_df['mean_horizontal'], label="horizontal")
plt.scatter(x=merged_df['queryFile'],y=merged_df['mean_vertical'], label="vertical")
plt.scatter(x=merged_df['queryFile'],y=merged_df['mean_combined'], label="combined")
plt.legend(loc="upper left")
plt.show()


#merged_df.plot.scatter(horizontal_df['mean_horizontal'])