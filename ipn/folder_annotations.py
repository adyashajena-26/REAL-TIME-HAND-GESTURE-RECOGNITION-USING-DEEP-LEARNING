import os

# Step 1: Read TXT file into a list of dictionaries
txt_file_path = 'Annot_List.txt'
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

# Assuming the first line contains column headers
headers = list(map(str.strip, lines[0].split(',')))
txt_data = [list(map(str.strip, line.split(','))) for line in lines[1:]]
txt_dict_list = [dict(zip(headers, values)) for values in txt_data]

# Step 2: List files in the folder
folder_path = '/Users/adyasha/Documents/research project/gesture-recognition-master/ipn/frames01'

# folder_files = [int(f.split('_')[-1][1:]) for f in os.listdir(folder_path) if f.split('_')[-1][1:].isdigit()]
folder_files = [f for f in os.listdir(folder_path) ]
# Step 3: Filter rows

filtered_data = [entry for entry in txt_dict_list if entry['video'] in folder_files]
print(filtered_data)

filtered_txt_path = 'filtered_data.txt'
with open(filtered_txt_path, 'w') as file:
    file.write('\t'.join(headers) + '\n')
    for entry in filtered_data:
        file.write('\t'.join(entry.values()) + '\n')
