import struct
import csv
 
# Path to your binary file
bin_file_path = 'model.bin'
 
# Path to the output CSV file
csv_file_path = 'output_file.csv'
 
# Read the binary file
with open(bin_file_path, 'rb') as bin_file:
    # Read the entire file into a bytes object
    bin_data = bin_file.read()
 
# Calculate how many floats are in the data
num_floats = len(bin_data) // 4  # size of a single float in bytes (4 for IEEE 754 single precision)
 
# Unpack the binary data to floats
# 'f' is the format for a single-precision float in little-endian
# '<' specifies little-endian byte order, 'num_floats * f' creates the correct format string
floats = struct.unpack('<' + 'f' * num_floats, bin_data)
 
# Write the floats to a CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for f in floats:
        writer.writerow([f])  # write each float to a new row
 
print("Conversion complete. The CSV file has been created.")
