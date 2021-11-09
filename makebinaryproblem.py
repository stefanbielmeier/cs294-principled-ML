import sys
import csv
import numpy as np
import pandas as pd

binaryset = []
class0 = 0
class1 = 0

if (len(sys.argv) == 1) or (len(sys.argv) > 3):
	print("Usage:")
	print(sys.argv[0]+" <filename.csv>")
	print("Creates a subset of the first two observed classes in a balanced multi-class dataset given in CSV file.")
	print("Note: This program only gives reliable results for balanced two-class problems.")
	sys.exit()

else:
    try: 
        data = pd.read_csv(sys.argv[1], delimiter=",").values
        labels = np.unique(data[:,-1])
        for row in range(data.shape[0]):
            if data[row,-1] == labels[0]:
                class0 += 1
                binaryset.append(data[row,:])
            if data[row,-1] == labels[1]:
                class1 += 1
                binaryset.append(data[row,:])
    except FileNotFoundError:
        print('File not found. Specify a path to an existing csv file.')

print("items in set", len(binaryset))
print("items in class 1:", class0)
print("items in class 2:", class1)


with open("binaryset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(binaryset)

print("wrote binaryset – done")