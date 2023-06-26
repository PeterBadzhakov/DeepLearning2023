# Python script to summarize dataset labels

import csv
import json
from collections import defaultdict


# Each book in dataset can have 0 or more labels
# Count occurances of each label


err_c = 0
avg_per_book = 0
def count_field_occurrences(csv_file, column_index):
    field_counts = defaultdict(int)

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) >= column_index:
                json_data = row[column_index - 1].strip()
                if json_data:
                    try:
                        json_dict = json.loads(json_data)
                        global avg_per_book
                        avg_per_book +=  len(json_dict.values())
                        for value in json_dict.values():
                            field_counts[value] += 1
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in row: {row[column_index - 1]}")
                else:
                    global err_c 
                    err_c += 1
                    print(f"No JSON data found in row: {row[:-1]}")
                    print(err_c)
        avg_per_book /= reader.line_num
        print(reader.line_num)
        print("AVG", avg_per_book)

    return dict(field_counts)

csv_file = 'booksummaries.txt'
column_index = 6

occurrences = count_field_occurrences(csv_file, column_index)
sorted = dict(sorted(occurrences.items(), key=lambda x: x[1]))
for value, count in sorted.items():
    print(f"Value: {value}, Count: {count}")
