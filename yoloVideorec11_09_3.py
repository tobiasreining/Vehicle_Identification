from ultralytics import YOLO
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # only if running on spyder on windows
import logging
import torch
print(torch.cuda.is_available())
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

dist = [21.95,36.45,26.08] #distances between known objects in the video 
tol = 6 # Tolerance in pixels
target_size = 288 #to save frames of equal size to preserve length info. ideally multiple of 32 for yolo
car_data={}
cropped_images={}
img_directory="cropped_images"
coords=[[[558,782],[891,633],[1207,817],[1405,646]],[[558,782],[1211,497],[1207,817],[1579,502]],[[890,631],[1211,497],[1400,650],[1579,502]]]
#top:[bottom left, top left, bottom right, top right],bottom:[bottom left, top left,bottom right,top
#right] 
#Slopes: 
slope_functions={"entry_x":{}, "entry_y": {}, "exit_x": {}, "exit_y": {}}
for idx,box in enumerate(coords):
    if box[0][1]==box[2][1] or box[0][0]==box[2][0]:
        print("Warning: Speed detection lines are parallel to coord. system")
        continue
    slope_entry_y=(box[3][1]-box[1][1])/(box[3][0]-box[1][0])
    slope_exit_y=(box[2][1]-box[0][1])/(box[2][0]-box[0][0])
    slope_entry_x=1/slope_entry_y
    slope_entry_x_alt=(box[3][0]-box[1][0])/(box[3][1]-box[1][1])
    slope_exit_x=1/slope_exit_y
    entry_function_y= lambda x, k=slope_entry_y,d=box[1][1],x0=box[1][0]: k*(x-x0)+d
    exit_function_y= lambda x, k=slope_exit_y,d=box[0][1],x0=box[0][0]: k*(x-x0)+d
    entry_function_x= lambda y, k=slope_entry_x,d=box[1][0],y0=box[1][1]: k*(y-y0)+d
    exit_function_x= lambda y, k=slope_exit_x,d=box[0][0],y0=box[0][1]: k*(y-y0)+d
    slope_functions["entry_y"][idx]=entry_function_y
    slope_functions["exit_y"][idx]=exit_function_y
    slope_functions["entry_x"][idx]=entry_function_x
    slope_functions["exit_x"][idx]=exit_function_x
    
    
entry_registered = {i: False for i in range(len(coords))}
exit_registered = {i: False for i in range(len(coords))}

model_path = 'best.pt'
model = YOLO(model_path)#, tracker='botsort.yaml')

print("Model device:", model.device)
model.to('cuda')
print("Model device after moving to GPU:", model.device)

video_path = 'rec_11_09_3.MOV'
cap = cv2.VideoCapture(video_path)
start_time_msec = 2 * 60 * 1000 + 5 * 1000
cap.set(cv2.CAP_PROP_POS_MSEC, start_time_msec)
#print(cv2.getBuildInformation())
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
else:
    print("Video file opened successfully.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break #ret==False: End of Video
    frameforscreencap = frame.copy()
    result = model.track(source=frame,persist=True, conf=0.05, iou=0.2, show=False, classes=[2,3,5,7])
    #conf: Confidence threshold for detection; iou: Intersection over union threshold
    result=result[0]
    boxes = result.boxes.xyxy
    ids = result.boxes.id

    # Draw the box outlines
    for coord in coords:
        cv2.line(frame, (coord[0][0],coord[0][1]), (coord[1][0], coord[1][1]), (0, 0, 255), 2)   
        cv2.line(frame, (coord[0][0],coord[0][1]), (coord[2][0], coord[2][1]), (0, 255, 115), 2) 
        cv2.line(frame, (coord[2][0],coord[2][1]), (coord[3][0], coord[3][1]), (0, 0, 0), 2)   
        cv2.line(frame, (coord[1][0],coord[1][1]), (coord[3][0], coord[3][1]), (255, 0, ), 2) 
    
    if boxes is None or ids is None:
        continue
    
    for box, track_id in zip(boxes, ids):
        track_id=int(track_id.item()) #convert tensor to native type
        x1, y1, x2, y2 = map(int, box[:4])
        midpoint = (x1 + x2) // 2, (y1 + y2) // 2
        y = midpoint[1] 
        x = midpoint[0]
        if track_id not in car_data:
            car_data[track_id]={}
        for idx, coord in enumerate(coords):
            box_data = car_data[track_id].get(idx,{})
            if abs(y2-slope_functions["entry_y"][idx](x1))<tol and x>coord[1][0] and x<coord[3][0] and "entry_time" not in box_data:
                cv2.line(frame, (coord[1][0], coord[1][1]), (coord[3][0], coord[3][1]), (100, 155, 100), 2)
                box_data["entry_time"]=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0                    

            elif abs(y2-slope_functions["exit_y"][idx](x1))<tol and x>coord[0][0] and x<coord[2][0] and "exit_time" not in box_data:
                cv2.line(frame, (coord[0][0], coord[0][1]), (coord[2][0], coord[2][1]), (100, 155, 100), 2)
                box_data["exit_time"] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if "entry_time" in box_data and "exit_time" in box_data:
                    speed = (3.6 * dist[idx]) / (box_data["exit_time"] - box_data["entry_time"])
                    print(f"Vehicle {track_id}: Speed in box {idx}: {speed:.2f} [km/h]")
                    box_data["Speed"]=speed
                if idx == 0: #take the picture at the top box end
                    width=(x2-x1)/target_size #normalized width. for later label creation
                    height=(y2-y1)/target_size
                    half_size=target_size/2
                    if width > 1 or height > 1:
                        half_size = half_size*2 # if vehicles are bigger than the box, double the box size
                        width=min(width/2,1)
                        height=min(height/2,1)
                    x1_new = int(max(x - half_size, 0))
                    x2_new = int(min(x + half_size, frameforscreencap.shape[1]))
                    y1_new = int(max(y - half_size, 0))
                    y2_new = int(min(y + half_size, frameforscreencap.shape[0]))
                    cropped_image = frameforscreencap[y1_new:y2_new, x1_new:x2_new]
                    cropped_images[track_id]=cropped_image #save it for now until the avg_speed is calculated
                    cv2.imshow('Cropped car', cropped_image)
                    base_name = os.path.basename(video_path)
                    file_name, file_extension = os.path.splitext(base_name)
                    save_path = f"cropped_images/{track_id}_{file_name}_wi{width:.2f}_he{height:.2f}.jpg"
                    if not os.path.exists(img_directory):
                        os.makedirs(img_directory)
                    cv2.imwrite(save_path, cropped_images[track_id])
                if idx == 1: # calculate avg speed at the last box
                    avg_speed=0
                    for i in car_data[track_id]:
                        speed = car_data[track_id][i].get("Speed")
                        if speed is not None:
                            avg_speed+=speed
                    avg_speed=avg_speed/len(car_data[track_id])
                    base_name = os.path.basename(video_path)
                    file_name, file_extension = os.path.splitext(base_name)
                    existing_filename=None
                    for filename in os.listdir(img_directory):
                        if filename.startswith(str(track_id)):
                            existing_filename = filename
                            break
                    if existing_filename:
                        new_save_path = f"cropped_images/{track_id}_{file_name}_{avg_speed:.2f}kmh_wi{width:.2f}_he{height:.2f}.jpg"
                        if not os.path.exists(new_save_path):
                            os.rename(os.path.join("cropped_images", existing_filename), new_save_path)
                        else:
                            print(f"File {new_save_path} already exists!")

            car_data[track_id][idx]=box_data
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) #rectangle on orig frame
        cv2.putText(frame, str(track_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


    cv2.imshow('Vehicle Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#%% 
print("\nCar Speeds (in km/h):")
print("-" * 50)

for car_id, data in car_data.items():
    speeds = [data[box_idx].get("Speed") for box_idx in range(len(coords))]  # This might give None if "Speed" doesn't exist for the box_idx
    valid_speeds = [speed for speed in speeds if isinstance(speed, (int, float))]
    average_speed = sum(valid_speeds) / len(valid_speeds) if valid_speeds else None
    speeds_str_list = [f"Box {i}: {speed:.2f}" for i, speed in enumerate(speeds) if isinstance(speed, (float, int))]
    if isinstance(average_speed, (float, int)):
        speeds_str_list.append(f"Average Speed: {average_speed:.2f}")
    speeds_str = ", ".join(speeds_str_list)
    if speeds_str:  # Only print if there are valid speeds for the car_id
        print(f"Car ID {car_id}: {speeds_str}")

print("-" * 50)

