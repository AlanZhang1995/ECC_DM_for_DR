import pandas as pd

# Load the CSV file
csv_file = "/home/haochen/Seagate8T/universe/DataSet/diabet_fundus/DDR-dataset/DR_grading/train_DDR.csv"  # Replace with your CSV file path
output_folder = "/home/haochen/Seagate8T/universe/DataSet/diabet_fundus/DDR-dataset/DR_grading/balanced_txt/"
data = pd.read_csv(csv_file)

# Separate data based on the Fold column
for fold_no in range(5):    
    fold_0 = data[data['Fold'] == fold_no]
    remaining_data = data[data['Fold'] != fold_no]
    print(fold_no, len(remaining_data), len(fold_0))

    # Save valid0.txt
    with open(output_folder+"valid{}.txt".format(fold_no), "w") as valid_file:
        for _, row in fold_0.iterrows():
            valid_file.write(f"{row['image']} {row['level']}\n")

    # Save train0.txt
    with open(output_folder+"train{}.txt".format(fold_no), "w") as train_file:
        for _, row in remaining_data.iterrows():
            train_file.write(f"{row['image']} {row['level']}\n")