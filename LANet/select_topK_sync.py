import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

low =  0.65 # 0.5/0.6/0.7
high = 0.95 # 0.8/0.9/1.0
num_of_C = 6

class_labels = {0:"NoDR", 1:"Mild", 2:"Mod", 3:"Sev", 4:"Prolif"}

def draw_hist(df):
    class_ids = df['class_id'].unique()
    plt.figure(figsize=(10, 6))

    for class_id in class_ids:
        #if not class_id in [1,3]:
        #    continue
        subset = df[df['class_id'] == class_id]
        plt.hist(subset['score']/5.0, bins=30, density=True, alpha=0.5, label=class_labels[class_id])

    plt.title('Score Distribution by Class')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig('score_hist.png')

def filter_and_save_top_rows(csv_file, output_txt, top_n=1500, sample_num=500, class_id='class_id', img_path='img_path'):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    print(len(df[df[class_id] == 1]), len(df[df[class_id] == 3]))
    # df = df[(df['score'] >= 0.65) & (df['score'] <= 0.95)]
    # df = df[(df['score'] >= low*num_of_C) & (df['score'] <= high*num_of_C)]
    print(len(df[df[class_id] == 1]), len(df[df[class_id] == 3]))
    # Select top 1500 rows if class_id == 1, and top 800 rows if class_id == 3
    # df_top = pd.concat([
    #     df[df['class_id'] == 1].sort_values(by='score', ascending=False).head(16000).sample(n=5000, random_state=42, replace=False),
    #     df[df['class_id'] == 3].sort_values(by='score', ascending=False).head(5000)
    # ])
    df_top = pd.concat([
        df[df[class_id] == 1].sample(n=5000, random_state=42, replace=False),
        df[df[class_id] == 3].sample(n=5000, random_state=42, replace=False),
        df[df[class_id] == 0].sample(n=5000, random_state=42, replace=False),
        df[df[class_id] == 2].sample(n=5000, random_state=42, replace=False),
        df[df[class_id] == 4].sample(n=5000, random_state=42, replace=False)
    ])

    draw_hist(df_top)
    
    # Extract the last part of 'img_path' after splitting by '/'
    df_top[img_path] = df_top[img_path].apply(lambda x: x.split('/')[-1])
    
    # Extract 'img_path' and 'class_id' columns
    df_top[[img_path, class_id]].to_csv(output_txt, sep=' ', index=False, header=False)
    
    print(f"Saved top {top_n} rows per class_id to {output_txt}")

# Example usage
csv_file = "/media/haochen/WD8TB/universe/DataSet/Retina_fundus/data.csv"  
#output_txt = "/media/haochen/WD8TB/universe/DataSet/Retina_fundus/sync_wo_selection.txt"  
output_txt = "/media/haochen/WD8TB/universe/DataSet/Retina_fundus/debug"  
# csv_file = "/home/haochen/Seagate8T/universe/DataSet/diabet_fundus/EyePacs/train_evaled.csv"  
# output_txt = "/home/haochen/Seagate8T/universe/DataSet/diabet_fundus/EyePacs/eyepacs_w_selection.txt"
filter_and_save_top_rows(csv_file, output_txt)



csv_file = "/media/haochen/WD8TB/universe/DataSet/Retina_genAvaCC_500K/data.csv"  
df = pd.read_csv(csv_file)
print(df.head())
draw_hist(df)