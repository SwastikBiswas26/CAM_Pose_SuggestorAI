import cv2
import json
import os
import time
from pose_engine import PoseDetector, get_pose_angles

def record_pose_tool():
    print("====================================================")
    print("        Pose Recorder Utility                       ")
    print("====================================================")
    print("How to use:")
    print("1. Stand in front of your camera.")
    print("2. Strike the pose you want to record.")
    print("3. Press 'S' to capture and save the pose.")
    print("4. Press 'Q' to quit without saving.")
    print("====================================================")

    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create target directory for poses if it doesn't exist
    os.makedirs(os.path.join("assets", "poses"), exist_ok=True)
    json_path = os.path.join("assets", "poses", "custom_poses.json")

    # Load existing custom poses
    custom_poses = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                custom_poses = json.load(f)
        except Exception as e:
            print(f"Error loading existing custom poses: {e}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Mirror frame for intuitive placement
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Detect pose
        detector.process_frame(frame)
        lm_list = detector.find_landmarks(frame, draw=True, draw_connections=True)

        cv2.putText(frame, "Press 'S' to Save | 'Q' to Quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Pose Recorder - Strike your pose!", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # Esc
            break
        elif key == ord('s') or key == ord('S'):
            if not lm_list:
                print("No pose detected! Please try again when your body is fully visible.")
                continue

            # Record pose details
            pose_name = input("Enter a name for this pose: ").strip()
            if not pose_name:
                print("Pose name cannot be empty. Cancelled.")
                continue

            description = input("Enter a brief description for this pose: ").strip()
            
            # Extract angles
            angles = get_pose_angles(lm_list)
            
            # Save normalized landmarks (x, y) for skeleton drawing template
            # Store only key points needed for drawing the skeleton guide
            landmarks_data = []
            for item in lm_list:
                landmarks_data.append({
                    "id": item[0],
                    "x": item[3], # norm_x
                    "y": item[4], # norm_y
                    "vis": item[6] # visibility
                })

            custom_poses[pose_name] = {
                "description": description or f"Custom pose: {pose_name}",
                "angles": angles,
                "landmarks": landmarks_data,
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            try:
                with open(json_path, 'w') as f:
                    json.dump(custom_poses, f, indent=4)
                print(f"Success! Pose '{pose_name}' has been recorded and saved to {json_path}")
            except Exception as e:
                print(f"Error saving pose: {e}")
            
            print("\nReturning to webcam stream... press 'Q' when finished.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_pose_tool()
