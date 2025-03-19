import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import glob

# Plotting mass point vs. boost values it has

# Get list of boosts from each available csv file
boosts = []

file_dir = str(sys.argv[1])
print(file_dir)
csv_files = glob.glob(f"{file_dir}/*.csv")

for arg in csv_files:
    tempboosts = []
    with open(arg) as csvfile:
        reader = csv.reader(csvfile)
        n = 0
        for row in reader:
            tempboosts.append(float(row[0]))
    boosts.append(tempboosts)

# Get mass of each file from the csv file name
def get_mass(csvfile, header):
    posind = csvfile.find(header) + len(header)
    if posind != -1:
        remaining_filename = csvfile[posind:].strip()
        pmass = remaining_filename.split('m')[0]
        mass = pmass.replace('p', '.')
        return mass
    else:
        return None

masses = []
for arg in csv_files:
    masses.append(float(get_mass(arg, "SAI_AtoGG_")))

# # Fix this so that it registers each mass point for the list of boosts if that makes sense
# for x, y in zip(boosts, masses):
#     plt.scatter(x, [y] * len(x))
# plt.xlabel('Boost')
# plt.ylabel('Mass (GeV)')
# plt.title('Distribution of Boost for each Mass Point')
# plt.yscale('symlog', linthresh=0.1)
# plt.yticks(masses, labels=[f"{val:.2f}" for val in masses])
# plt.show()

flat_boosts = [item for sublist in boosts for item in sublist]

plt.figure(figsize=(8, 5))
# plt.hist(flat_boosts, bins=30, edgecolor='black', alpha=0.7)  # Adjust bins for resolution
plt.hist(boosts, bins=10, edgecolor='black', alpha=0.7)  # Adjust bins for resolution
plt.xlabel("Boost")
plt.ylabel("Frequency")
plt.title("Histogram of Boosts")
plt.grid(True)
plt.show()

