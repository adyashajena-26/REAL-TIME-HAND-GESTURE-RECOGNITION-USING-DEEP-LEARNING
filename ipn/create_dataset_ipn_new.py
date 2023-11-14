import os
import mediapipe as mp
import cv2
import numpy as np
import time
from tqdm import tqdm

# Read the CSV file
txt_file_path = 'filtered_data.txt'
folder_path = 'frames01'
with open(txt_file_path, 'r') as file:
    lines = file.readlines()
data =[]
# Assuming the first line contains column headers
headers = list(map(str.strip, lines[0].split(',')))
frame_dict = {}
with open(txt_file_path, 'r') as file:

    # Skip the header
    header = file.readline()
    
    # Iterate through the remaining lines
    for line in file:
        # Split the line into an array
        row = line.strip().split('\t')
        frame_key = str(row[0].split('_')[-1][1:])
        if frame_key in frame_dict:
             frame_dict[frame_key].append(row)
        else:
            frame_dict[frame_key] = [row]

        
        # Convert elements to appropriate data types if needed
        # For example, if you want to convert numeric values to integers:
        # row = [int(x) if x.isdigit() else x for x in row]
        
        # Append the row to the data list
        data.append(row)

# Define the desired sequence length
sequence_length = 30

folder_files = [f for f in os.listdir(folder_path) ]
all_file_paths = []
frame_data = []

for image_folder in folder_files:
    if image_folder == '.DS_Store':
        continue
    frame_key = image_folder.split("_")[-1][1:]
    frame_data.append(frame_dict[frame_key])
    for filename in os.listdir(os.path.join(folder_path,image_folder)):
        if filename.lower().endswith(('.jpg')):
            # Construct the full path to the image file
            file_path = os.path.join(folder_path, image_folder, filename)
            all_file_paths.append(file_path)
            # Now, you can perform operations on the image file using 'file_path'

action_data_list = []
for action_data in frame_data:

    for action in action_data:
        action_frame=[]
        for i in range(int(action[3]),int(action[4])+1):
            formatted_number = str(i).zfill(6)
            action_frame.append(action[0]+"_"+str(formatted_number))
        action_frame.append(action[2])
        action_data_list.append(action_frame)
        
# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)    

data_dict = {}
for sample in tqdm(action_data_list, desc="Processing samples"):
    idx = sample[-1]
    print(data_dict)
    if idx == '1' or idx=='2' or idx=='3':
        continue
    sample = sample[:-1]
    if data_dict!={} and all(value == 20 for value in data_dict.values()):
        print("desired data obtained")
        break
    if idx in data_dict and data_dict[idx]==20:
        print(f'class {idx} complete')
        continue
   
    
    data = []
    for i in range(len(sample)):
        img = cv2.imread(os.path.join('frames01',sample[i][0:-7],sample[i]+".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        created_time = int(time.time())

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                angle_label = np.array([angle], dtype=np.float32)
                angle_label = np.append(angle_label, idx)

                d = np.concatenate([joint.flatten(), angle_label])

                data.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break

    data = np.array(data)
    print(idx, data.shape)
    full_seq_data = []
    if len(data)==0:
        continue
    if len(data)>sequence_length:
        for seq in range(len(data) - sequence_length):
            full_seq_data.append(data[seq:seq + sequence_length])
    else:
        repeat_factor = sequence_length // len(data)
        repeated_array = np.tile(data, (repeat_factor, 1))

        remaining_rows = sequence_length % len(data)
        full_seq_data = np.concatenate((repeated_array, data[:remaining_rows, :]), axis=0)
        full_seq_data = full_seq_data[np.newaxis, :, :]
    full_seq_data = np.array(full_seq_data)
    print(sample[i][0:-7],full_seq_data.shape)

    if idx in data_dict and data_dict[idx]<=20:
        np.save(os.path.join('dataset_new', f'seq_{idx}_{sample[i][0:-7]}_{created_time}'), full_seq_data)
        data_dict[idx]+=1
    elif idx not in data_dict: 
        np.save(os.path.join('dataset_new', f'seq_{idx}_{sample[i][0:-7]}_{created_time}'), full_seq_data)
    
        data_dict[idx]=1
       


    
cv2.destroyAllWindows()


