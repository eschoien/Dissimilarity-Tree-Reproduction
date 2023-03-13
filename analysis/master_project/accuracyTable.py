import json
from scripts.prettytable import PrettyTable
from datetime import timedelta

# Run from DTR directory, will not find files if cd'ed into folder of this file
def calculate_accuracy(data, k):

    correct_guesses = 0

    for query in data['results']:
        if query['queryFileID'] in query['bestMatches'][:k]:
            correct_guesses += 1

    amount = str(correct_guesses) + '/' + str(len(data['results']))
    accuracy = (correct_guesses / len(data['results'])) * 100
    return amount, accuracy

def calculate_time(data):
    time = 0.0
    for query in data['results']:
        time += query['executionTimeSeconds']

    avg_time = time / len(data['results'])
    
    return time, avg_time




# --- ARGUMENTS ---
basePath = 'output/lsh/measurements/v2/'
ks = [1, 3, 5, 10]
permutations = [10, 50, 100]
thresholds =  ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]
limits = [100, 500, 1000, 0, 101, 501, 1001]
# -----------------


for k in ks:
    for p in permutations:
        try:
            timeSec = 0.0
            outputTable = PrettyTable(['Jaccard', "P100", "P500", "P1000", "-", "C100", "C500", "C1000"])
            outputTable.align = "r"

            accuracies = {}
            times = {}

            for j in thresholds:
                times[j] = {}
                accuracies[j] = {}
                partial = True
                for d in limits:
                    try:
                        if d == 0:
                            partial = False

                        if partial:
                            signature_data = json.load(open(basePath + 'partial_objects/permcount'+str(p)+'/measurement-'+j+'-'+str(d)+'-'+str(p)+'.json'))
                        else:
                            signature_data = json.load(open(basePath + 'complete_objects/permcount'+str(p)+'/measurement-'+j+'-'+str(d-1)+'-'+str(p)+'.json'))

                        # TIME
                        time, avg_time = calculate_time(signature_data)
                        times[j][d] = time
                        timeSec += time

                        # ACCURACY
                        signature_accuracy = calculate_accuracy(signature_data, k)
                        accuracies[j][d] = f'{round(signature_accuracy[1],2)}%'
                       #accuracies[j][d] = f'{round(signature_accuracy[1],2)}%, {round(time, 3)}' # Also print avg execution time
                       #accuracies[j][d] = f'{signature_accuracy[0]}' # Print number of correct results / number of queries

                    except:
                        accuracies[j][d] = "-"
                
                outputTable.add_row([j] + list(accuracies[j].values()))

            ## ------------ ADD TIMES FOR EACH COLUMN -----------------------
            total_times = []
            for d in limits:
                
                total_time = 0.0
                
                if d != 0:
                    for j in thresholds:
                        total_time += times[j][d]
                    total_times.append(timedelta(seconds=round(total_time)))
                else:
                    total_times.append('-')
            
            outputTable.add_row(['Total time'] + total_times)
            ## -------------------------------------------------------------
            

            #print("             {-----------Partial-----------}   {-----------Complete----------}")
            print(f'Results for top-k={k}, Permutations p={p}, Total time: {timedelta(seconds=round(timeSec))}')
            print(outputTable)
            
        except: # If the specified measurement files doesn't exist

            print(f'Results for top-k={k}, Permutations p={p}, Total time: ...')
            print(f'--- Missing data for Permutations p={p} ---')
            print()

        finally: #Add spacing in both cases
            print()
            print()