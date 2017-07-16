import csv
import sys
from itertools import groupby
import csv
import json

if __name__ == "__main__":

    with open("myPoC.csv", 'rU') as f_obj:
        reader = csv.DictReader(f_obj, delimiter=',')
	
	dict={};
	L = [];
	N=[];
	tmp={};
	with open("poc.json", "wb") as fs:
		for line in reader:	
			if line["source"]=="Serbia and Kosovo (S/RES/1244 (1999))":
				line["source"]= "Serbia and Kosovo";
			if line["target"] =="Serbia and Kosovo (S/RES/1244 (1999))":
				line["target"] = "Serbia and Kosovo";
			
			if line["value"]!="*":
				L.append({"Year":line["Year"],  "source": line["source"] , "target": line["target"], "value": line["value"]});
		
		tmpN=[];
		for i in range(len(L)):
			tmpN.append(L[i]["source"]);
			tmpN.append(L[i]["target"]);
		tmpUnique = list(set(tmpN));
		tmpUnique.sort();
		for item in tmpUnique:
			N.append({"name": item})
		
		dict = {"links": L , "nodes": N}
	
		json.dump(dict,fs);
		
