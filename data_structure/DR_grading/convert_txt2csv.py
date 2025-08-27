import csv

# Define file paths
input_txt_file = 'train.txt'  # Replace with your input text file path
output_csv_file = 'train.csv'  # Replace with your desired CSV file path

# Read the text file and write to the CSV file
with open(input_txt_file, 'r') as txt_file, open(output_csv_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image', 'level'])
    for line in txt_file:
        # Split the line into columns and write to CSV
        csv_writer.writerow(line.strip().split())

print(f"Data has been saved to {output_csv_file}")
