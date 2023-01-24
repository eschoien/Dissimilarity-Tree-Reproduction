import json

def calculate_accuracy(data):

    correct_guesses = 0
    total_guesses = 0

    for i in data['results']:
        total_guesses += 1
        if i['queryFileID'] == i['bestSearchResultFileID']:
            correct_guesses += 1

    accuracy = (correct_guesses / total_guesses) * 100
    return accuracy

f = open('query_time_results/combined_measurements_indexed.json')
combined_data = json.load(f)

f = open('query_time_results/horizontal_measurements_indexed.json')
horizontal_data = json.load(f)

f = open('query_time_results/vertical_measurements_indexed.json')
vertical_data = json.load(f)

f = open('query_time_results/figure10_indexed_search_100000.json')
author_data = json.load(f)

horizontal_accuracy = calculate_accuracy(horizontal_data)
vertical_accuracy = calculate_accuracy(vertical_data)
combined_accuracy = calculate_accuracy(combined_data)
author_accuracy = calculate_accuracy(author_data)

print(f'Accuracy from horizontal: {horizontal_accuracy}%')
print(f'Accuracy from vertical: {vertical_accuracy}%')
print(f'Accuracy from combined: {combined_accuracy}%')
print(f'Accuracy from author: {author_accuracy}%')