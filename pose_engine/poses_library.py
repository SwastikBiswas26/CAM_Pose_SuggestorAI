import os
import json

# Pre-defined built-in poses with target angles
BUILTIN_POSES = {
    "Casual Standing (Left Hand on Hip)": {
        "description": "Stand straight, place your left hand on your hip, and let your right arm hang casually down.",
        "angles": {
            "left_elbow": 100.0,
            "right_elbow": 170.0,
            "left_shoulder": 35.0,
            "right_shoulder": 15.0,
            "left_hip": 175.0,
            "right_hip": 175.0,
            "left_knee": 175.0,
            "right_knee": 175.0
        },
        # Standard landmarks template (normalized x, y)
        "landmarks": [
            {"id": 0, "x": 0.5, "y": 0.15, "vis": 0.99},   # Nose
            {"id": 11, "x": 0.43, "y": 0.28, "vis": 0.99}, # Left shoulder
            {"id": 12, "x": 0.57, "y": 0.28, "vis": 0.99}, # Right shoulder
            {"id": 13, "x": 0.35, "y": 0.38, "vis": 0.99}, # Left elbow (bent)
            {"id": 14, "x": 0.60, "y": 0.44, "vis": 0.99}, # Right elbow (straight)
            {"id": 15, "x": 0.42, "y": 0.44, "vis": 0.99}, # Left wrist (on hip)
            {"id": 16, "x": 0.62, "y": 0.60, "vis": 0.99}, # Right wrist (down)
            {"id": 23, "x": 0.45, "y": 0.55, "vis": 0.99}, # Left hip
            {"id": 24, "x": 0.55, "y": 0.55, "vis": 0.99}, # Right hip
            {"id": 25, "x": 0.45, "y": 0.75, "vis": 0.99}, # Left knee
            {"id": 26, "x": 0.55, "y": 0.75, "vis": 0.99}, # Right knee
            {"id": 27, "x": 0.45, "y": 0.92, "vis": 0.99}, # Left ankle
            {"id": 28, "x": 0.55, "y": 0.92, "vis": 0.99}  # Right ankle
        ]
    },
    "Arms Crossed": {
        "description": "Cross your arms over your chest, stand tall with shoulder-width stance.",
        "angles": {
            "left_elbow": 75.0,
            "right_elbow": 75.0,
            "left_shoulder": 40.0,
            "right_shoulder": 40.0,
            "left_hip": 175.0,
            "right_hip": 175.0,
            "left_knee": 175.0,
            "right_knee": 175.0
        },
        "landmarks": [
            {"id": 0, "x": 0.5, "y": 0.15, "vis": 0.99},
            {"id": 11, "x": 0.42, "y": 0.28, "vis": 0.99},
            {"id": 12, "x": 0.58, "y": 0.28, "vis": 0.99},
            {"id": 13, "x": 0.35, "y": 0.36, "vis": 0.99},
            {"id": 14, "x": 0.65, "y": 0.36, "vis": 0.99},
            {"id": 15, "x": 0.54, "y": 0.36, "vis": 0.99}, # Crossed left wrist
            {"id": 16, "x": 0.46, "y": 0.36, "vis": 0.99}, # Crossed right wrist
            {"id": 23, "x": 0.45, "y": 0.55, "vis": 0.99},
            {"id": 24, "x": 0.55, "y": 0.55, "vis": 0.99},
            {"id": 25, "x": 0.45, "y": 0.75, "vis": 0.99},
            {"id": 26, "x": 0.55, "y": 0.75, "vis": 0.99},
            {"id": 27, "x": 0.45, "y": 0.92, "vis": 0.99},
            {"id": 28, "x": 0.55, "y": 0.92, "vis": 0.99}
        ]
    },
    "Hand on Chin (The Thinker)": {
        "description": "Raise your right hand to touch your chin with your right elbow bent, resting on your crossed left arm.",
        "angles": {
            "left_elbow": 80.0,
            "right_elbow": 45.0,
            "left_shoulder": 35.0,
            "right_shoulder": 75.0,
            "left_hip": 170.0,
            "right_hip": 170.0,
            "left_knee": 175.0,
            "right_knee": 175.0
        },
        "landmarks": [
            {"id": 0, "x": 0.5, "y": 0.15, "vis": 0.99},
            {"id": 11, "x": 0.42, "y": 0.28, "vis": 0.99},
            {"id": 12, "x": 0.58, "y": 0.28, "vis": 0.99},
            {"id": 13, "x": 0.35, "y": 0.38, "vis": 0.99},
            {"id": 14, "x": 0.55, "y": 0.38, "vis": 0.99},
            {"id": 15, "x": 0.48, "y": 0.38, "vis": 0.99},
            {"id": 16, "x": 0.50, "y": 0.20, "vis": 0.99}, # Right hand on chin
            {"id": 23, "x": 0.45, "y": 0.55, "vis": 0.99},
            {"id": 24, "x": 0.55, "y": 0.55, "vis": 0.99},
            {"id": 25, "x": 0.45, "y": 0.75, "vis": 0.99},
            {"id": 26, "x": 0.55, "y": 0.75, "vis": 0.99},
            {"id": 27, "x": 0.45, "y": 0.92, "vis": 0.99},
            {"id": 28, "x": 0.55, "y": 0.92, "vis": 0.99}
        ]
    },
    "Model Pose (Hand in Pocket)": {
        "description": "Slightly tilt your body, slide your right hand into your pocket, bent at the elbow. Keep your left hand relaxed.",
        "angles": {
            "left_elbow": 160.0,
            "right_elbow": 115.0,
            "left_shoulder": 15.0,
            "right_shoulder": 25.0,
            "left_hip": 175.0,
            "right_hip": 165.0,
            "left_knee": 175.0,
            "right_knee": 165.0
        },
        "landmarks": [
            {"id": 0, "x": 0.5, "y": 0.15, "vis": 0.99},
            {"id": 11, "x": 0.43, "y": 0.28, "vis": 0.99},
            {"id": 12, "x": 0.57, "y": 0.28, "vis": 0.99},
            {"id": 13, "x": 0.38, "y": 0.44, "vis": 0.99},
            {"id": 14, "x": 0.65, "y": 0.38, "vis": 0.99},
            {"id": 15, "x": 0.38, "y": 0.60, "vis": 0.99},
            {"id": 16, "x": 0.55, "y": 0.50, "vis": 0.99}, # Right hand in pocket
            {"id": 23, "x": 0.45, "y": 0.55, "vis": 0.99},
            {"id": 24, "x": 0.55, "y": 0.55, "vis": 0.99},
            {"id": 25, "x": 0.44, "y": 0.75, "vis": 0.99},
            {"id": 26, "x": 0.56, "y": 0.74, "vis": 0.99},
            {"id": 27, "x": 0.44, "y": 0.92, "vis": 0.99},
            {"id": 28, "x": 0.56, "y": 0.90, "vis": 0.99}
        ]
    },
    "Victory V-Sign (Hands Up)": {
        "description": "Raise both of your arms high, forming a 'V' shape, celebrating victory!",
        "angles": {
            "left_elbow": 165.0,
            "right_elbow": 165.0,
            "left_shoulder": 140.0,
            "right_shoulder": 140.0,
            "left_hip": 175.0,
            "right_hip": 175.0,
            "left_knee": 175.0,
            "right_knee": 175.0
        },
        "landmarks": [
            {"id": 0, "x": 0.5, "y": 0.20, "vis": 0.99},
            {"id": 11, "x": 0.43, "y": 0.38, "vis": 0.99},
            {"id": 12, "x": 0.57, "y": 0.38, "vis": 0.99},
            {"id": 13, "x": 0.28, "y": 0.23, "vis": 0.99}, # Left elbow raised
            {"id": 14, "x": 0.72, "y": 0.23, "vis": 0.99}, # Right elbow raised
            {"id": 15, "x": 0.18, "y": 0.10, "vis": 0.99}, # Left wrist high
            {"id": 16, "x": 0.82, "y": 0.10, "vis": 0.99}, # Right wrist high
            {"id": 23, "x": 0.45, "y": 0.65, "vis": 0.99},
            {"id": 24, "x": 0.55, "y": 0.65, "vis": 0.99},
            {"id": 25, "x": 0.45, "y": 0.80, "vis": 0.99},
            {"id": 26, "x": 0.55, "y": 0.80, "vis": 0.99},
            {"id": 27, "x": 0.45, "y": 0.95, "vis": 0.99},
            {"id": 28, "x": 0.55, "y": 0.95, "vis": 0.99}
        ]
    }
}

def get_all_poses():
    """
    Loads custom poses from JSON and merges them with builtin poses.
    """
    all_poses = BUILTIN_POSES.copy()
    custom_json_path = os.path.join("assets", "poses", "custom_poses.json")
    
    if os.path.exists(custom_json_path):
        try:
            with open(custom_json_path, 'r') as f:
                custom_poses = json.load(f)
                # Merge custom poses (custom poses will overwrite builtin if name matches)
                for pose_name, pose_data in custom_poses.items():
                    all_poses[pose_name] = pose_data
        except Exception as e:
            print(f"Error loading custom poses: {e}")
            
    return all_poses

# For legacy backwards compatibility
POSES_LIBRARY = BUILTIN_POSES
