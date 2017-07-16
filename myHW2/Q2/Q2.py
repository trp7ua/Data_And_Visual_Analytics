import csv
import sys
from itertools import groupby
import csv


if __name__ == "__main__":

    with open("nepal.csv") as f_obj:
        reader = csv.DictReader(f_obj, delimiter=',')

	path = "nepal_new.csv"
	newRow=["District",2007,2008,2009,2010,2011]
	with open(path, "wb") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		writer.writerow(newRow)	
		
		for key, group in groupby(reader, lambda x: x["District"]):	
			newRow =[];
			for thing in group:
				newRow.append(thing["Total"])
			newRow.insert(0,thing["District"])
		
			writer.writerow(newRow)		
			