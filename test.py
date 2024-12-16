from file_reader import read_file
from pprint import pprint

file_path = "output/Temperature Sweep/Sample2.ngb-dm2.csv"
assay = "Temperature Sweep"

metadata, experiments = read_file(file_path, assay, debug=True)

print("General Metadata:", metadata)
pprint(experiments)
