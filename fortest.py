import csv
import re

with open('women.csv') as namescsv:
    namereader = csv.reader(namescsv)
    for row in namereader:
        for cell in row:
            cell = re.sub(r'[^\w=]', '',cell)
            print cell
