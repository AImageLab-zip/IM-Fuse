import csv

# Input and output file paths
input_file = "/homes/ocarpentiero/IM-Fuse/D2Net/datalist/train.txt"
output_file = "/homes/ocarpentiero/IM-Fuse/D2Net/datalist/train15splits.csv"

# Read all case IDs from the input file
with open(input_file, "r") as f:
    cases = [line.strip() for line in f if line.strip()]

# Write to CSV with header
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["case", "mask"])
    for case in cases:
        writer.writerow([case, str([True,True,True,True])])

print(f"CSV written to {output_file}")
