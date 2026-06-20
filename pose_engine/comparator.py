import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the 2D angle (in degrees) at vertex B formed by points A, B, and C.
    Each point should be [x, y] coordinates.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (vertex)
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

def get_pose_angles(lm_list):
    """
    Extracts the 8 key joint angles from a list of landmarks.
    lm_list format: each item is [id, px_x, px_y, norm_x, norm_y, norm_z, visibility]
    We use normalized x, y coordinates for calculation to make it resolution-independent.
    """
    if not lm_list or len(lm_list) < 33:
        return {}

    # Extract [x, y] coordinates for required landmarks
    points = {}
    for item in lm_list:
        idx = item[0]
        points[idx] = [item[3], item[4]] # norm_x, norm_y

    angles = {}

    try:
        # 1. Left Elbow: Shoulder(11) -> Elbow(13) -> Wrist(15)
        angles['left_elbow'] = calculate_angle(points[11], points[13], points[15])
        
        # 2. Right Elbow: Shoulder(12) -> Elbow(14) -> Wrist(16)
        angles['right_elbow'] = calculate_angle(points[12], points[14], points[16])
        
        # 3. Left Shoulder: Hip(23) -> Shoulder(11) -> Elbow(13)
        angles['left_shoulder'] = calculate_angle(points[23], points[11], points[13])
        
        # 4. Right Shoulder: Hip(24) -> Shoulder(12) -> Elbow(14)
        angles['right_shoulder'] = calculate_angle(points[24], points[12], points[14])
        
        # 5. Left Hip: Shoulder(11) -> Hip(23) -> Knee(25)
        angles['left_hip'] = calculate_angle(points[11], points[23], points[25])
        
        # 6. Right Hip: Shoulder(12) -> Hip(24) -> Knee(26)
        angles['right_hip'] = calculate_angle(points[12], points[24], points[26])
        
        # 7. Left Knee: Hip(23) -> Knee(25) -> Ankle(27)
        angles['left_knee'] = calculate_angle(points[23], points[25], points[27])
        
        # 8. Right Knee: Hip(24) -> Knee(26) -> Ankle(28)
        angles['right_knee'] = calculate_angle(points[24], points[26], points[28])

    except KeyError:
        # Landmarks not complete
        pass

    return angles

def get_feedback(user_angles, target_angles, tolerance=20.0):
    """
    Compares user angles with target angles.
    Returns:
      - score: overall matching score (0 to 100)
      - feedback_list: list of strings with specific feedback details
      - status_dict: status (OK, Close, Adjust) for each joint
    """
    feedback_list = []
    status_dict = {}
    total_score = 0.0
    joints_compared = 0

    # Friendly joint names for messages
    joint_names = {
        'left_elbow': 'left elbow',
        'right_elbow': 'right elbow',
        'left_shoulder': 'left shoulder',
        'right_shoulder': 'right shoulder',
        'left_hip': 'left hip',
        'right_hip': 'right hip',
        'left_knee': 'left knee',
        'right_knee': 'right knee'
    }

    for joint, target_val in target_angles.items():
        if joint not in user_angles:
            continue

        user_val = user_angles[joint]
        diff = user_val - target_val
        abs_diff = abs(diff)

        # Joint score: linear decay. 0 points at 45 degrees mismatch or more
        joint_score = max(0.0, 100.0 - (abs_diff / 45.0) * 100.0)
        total_score += joint_score
        joints_compared += 1

        # Determine status and feedback
        if abs_diff <= tolerance:
            status_dict[joint] = 'GOOD'
        elif abs_diff <= tolerance * 2:
            status_dict[joint] = 'CLOSE'
            # Give general feedback
            if diff > 0:
                feedback_list.append(f"Adjust your {joint_names[joint]} slightly")
            else:
                feedback_list.append(f"Adjust your {joint_names[joint]} slightly")
        else:
            status_dict[joint] = 'ADJUST'
            # Provide direction for adjustment depending on the joint type
            if 'elbow' in joint or 'knee' in joint or 'hip' in joint:
                if diff > 0:
                    feedback_list.append(f"Bend your {joint_names[joint]} more")
                else:
                    feedback_list.append(f"Straighten your {joint_names[joint]} more")
            elif 'shoulder' in joint:
                if diff > 0:
                    feedback_list.append(f"Lower your {joint_names[joint]} arm")
                else:
                    feedback_list.append(f"Raise your {joint_names[joint]} arm")

    overall_score = (total_score / joints_compared) if joints_compared > 0 else 0.0
    
    # If feedback list is empty, user is in perfect pose!
    if not feedback_list and overall_score >= 85:
        feedback_list.append("Pose matches perfectly! Click the photo now!")

    return int(overall_score), feedback_list, status_dict
