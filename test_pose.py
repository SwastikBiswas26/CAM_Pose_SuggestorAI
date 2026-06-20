import cv2
import numpy as np
from pose_engine import PoseDetector, get_pose_angles, get_feedback, POSES_LIBRARY

def test_pipeline():
    print("Testing imports and classes...")
    try:
        detector = PoseDetector()
        print("[SUCCESS] PoseDetector initialized successfully.")
    except Exception as e:
        print(f"[FAILURE] PoseDetector initialization failed: {e}")
        return False

    print("Checking POSES_LIBRARY...")
    if POSES_LIBRARY:
        print(f"[SUCCESS] Loaded {len(POSES_LIBRARY)} reference poses from library.")
        for pose_name, data in POSES_LIBRARY.items():
            print(f"  - {pose_name}: {data['description']}")
    else:
        print("[FAILURE] Poses library is empty or failed to load.")
        return False

    print("Running dummy frame detection...")
    try:
        # Create a black image
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame
        detector.process_frame(dummy_img)
        
        # Get landmarks
        lm_list = detector.find_landmarks(dummy_img, draw=True)
        print(f"[SUCCESS] Dummy frame processed. Detected landmarks: {len(lm_list)}")
        
        # Test comparator on empty/dummy landmarks
        user_angles = get_pose_angles(lm_list)
        target_pose_name = list(POSES_LIBRARY.keys())[0]
        target_angles = POSES_LIBRARY[target_pose_name]["angles"]
        score, feedback, status = get_feedback(user_angles, target_angles)
        print(f"[SUCCESS] Comparator run. Score: {score}, Feedback count: {len(feedback)}")

    except Exception as e:
        print(f"[FAILURE] Process or comparator failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("Starting Pose engine tests...")
    if test_pipeline():
        print("\nAll backend components are working correctly and ready!")
    else:
        print("\nSome tests failed. Check dependencies.")
