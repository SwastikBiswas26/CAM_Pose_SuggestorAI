import cv2
import numpy as np
import streamlit as st
import os
import time
import glob
import io
import hashlib
import json
from PIL import Image
from pose_engine import PoseDetector, get_pose_angles, get_feedback, POSES_LIBRARY, get_all_poses

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Pose Suggestion & Camera",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Dark Modern Aesthetic
st.markdown("""
    <style>
        /* Modern font and gradients */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .main-header {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(45deg, #00FFCC, #0077FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subheader {
            font-size: 1.2rem;
            color: #888888;
            margin-bottom: 2rem;
        }
        
        .card {
            background-color: #1E1E24;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #2D2D35;
            margin-bottom: 1rem;
        }
        
        .feedback-good {
            color: #00FFCC;
            font-weight: 600;
        }
        .feedback-close {
            color: #FFCC00;
            font-weight: 600;
        }
        .feedback-adjust {
            color: #FF3366;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Ensure Capture Directories Exist
os.makedirs(os.path.join("assets", "captures"), exist_ok=True)
os.makedirs(os.path.join("assets", "poses"), exist_ok=True)

# Initialize Session State
if 'captured_image_info' not in st.session_state:
    st.session_state.captured_image_info = None
if 'processed_state_key' not in st.session_state:
    st.session_state.processed_state_key = None

def generate_silhouette_preview(landmarks_template):
    """
    Creates a black canvas with the target pose skeleton drawn on it to show in the UI.
    """
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    if not landmarks_template:
        # Draw a question mark or simple box if no landmarks
        cv2.putText(canvas, "?", (135, 170), cv2.FONT_HERSHEY_SIMPLEX, 4, (100, 100, 100), 4)
        return canvas
    
    # Simple mock drawing of connections for preview
    from pose_engine.detector import POSE_CONNECTIONS
    h, w, c = canvas.shape
    pts = {}
    
    for lm in landmarks_template:
        idx = lm["id"]
        # Scale to 300x300 canvas
        pts[idx] = (int(lm["x"] * w), int(lm["y"] * h))

    # Draw lines
    for conn in POSE_CONNECTIONS:
        s, e = conn
        if s in pts and e in pts:
            cv2.line(canvas, pts[s], pts[e], (0, 255, 204), 2, lineType=cv2.LINE_AA)
            
    # Draw points
    for idx, pt in pts.items():
        cv2.circle(canvas, pt, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        
    return canvas

# Main Web App UI
def main():
    st.markdown('<div class="main-header">PoseGenie AI 📸</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Strike the perfect pose with real-time AI guidance and feedback</div>', unsafe_allow_html=True)
    
    # Load all poses (built-in and recorded)
    poses = get_all_poses()
    
    # Sidebar
    st.sidebar.title("Pose Library")
    selected_pose_name = st.sidebar.selectbox("Choose a target pose:", list(poses.keys()))
    pose_data = poses[selected_pose_name]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### **Pose Details**")
    st.sidebar.markdown(f"**Description:** {pose_data['description']}")
    
    # Render guide skeleton preview in Sidebar
    st.sidebar.markdown("### **Skeleton Guide Outline**")
    preview_img = generate_silhouette_preview(pose_data.get("landmarks", []))
    st.sidebar.image(preview_img, channels="BGR", width="stretch")
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Tip: Stand far enough so your full body (head to ankles) is visible in the camera frame.")
    
    # Tab navigation
    tab_guide, tab_gallery, tab_create = st.tabs(["🎯 Pose Guide & Camera", "🖼️ My Photo Gallery", "➕ Create Custom Pose"])
    
    with tab_guide:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### How it works:")
            st.markdown("""
                1. Select your target pose from the sidebar.
                2. Choose your input method below (Webcam or Upload).
                3. Strike the pose and capture/upload.
                4. The AI will overlay the guide, detect your posture, and show suggestions!
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            input_method = st.radio("Choose Input Method:", ["📷 Use Browser Camera", "📤 Upload Photo"], key="input_method")
            
            input_file = None
            if input_method == "📷 Use Browser Camera":
                input_file = st.camera_input("Strike your pose!")
            else:
                input_file = st.file_uploader("Upload an image containing the pose", type=["png", "jpg", "jpeg"])
                
            if input_file is not None:
                file_bytes = input_file.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()
                current_state_key = (file_hash, selected_pose_name)
                
                # Run detection if file is new or target pose changed
                if st.session_state.processed_state_key != current_state_key:
                    with st.spinner("Analyzing pose..."):
                        try:
                            # 1. Load image and convert to BGR (OpenCV format)
                            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                            img_array = np.array(image)
                            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                            
                            # 2. Detect landmarks
                            detector = PoseDetector()
                            detector.process_frame(img_bgr)
                            lm_list = detector.find_landmarks(img_bgr, draw=False)
                            
                            # 3. Calculate score & feedback suggestions
                            target_angles = pose_data["angles"]
                            score = 0
                            suggestions = []
                            status_dict = {}
                            
                            if lm_list:
                                user_angles = get_pose_angles(lm_list)
                                score, suggestions, status_dict = get_feedback(user_angles, target_angles)
                            else:
                                suggestions = ["No pose detected. Make sure your full body (head to ankles) is visible in the frame."]
                            
                            # 4. Generate clean and analyzed image copies
                            img_clean = img_bgr.copy()
                            img_analysed = img_bgr.copy()
                            
                            # Draw Target guide outline template
                            detector.draw_guide_silhouette(img_analysed, pose_data.get("landmarks", []), color=(150, 150, 150), thickness=2)
                            
                            if lm_list:
                                # Determine skeleton color
                                if score >= 80:
                                    skeleton_color = (0, 255, 0)       # Green (Good)
                                elif score >= 50:
                                    skeleton_color = (0, 255, 255)     # Yellow (Close)
                                else:
                                    skeleton_color = (0, 0, 255)       # Red (Adjust)
                                    
                                detector.find_landmarks(img_analysed, draw=True, draw_connections=True, color=skeleton_color)
                            
                            # 5. Save files locally
                            timestamp = int(time.time())
                            raw_path = f"assets/captures/raw_{timestamp}.png"
                            analysed_path = f"assets/captures/analysed_{timestamp}.png"
                            
                            cv2.imwrite(raw_path, img_clean)
                            cv2.imwrite(analysed_path, img_analysed)
                            
                            # 6. Save results to Session State
                            st.session_state.captured_image_info = {
                                "timestamp": timestamp,
                                "pose_name": selected_pose_name,
                                "score": score,
                                "suggestions": suggestions,
                                "status_dict": status_dict,
                                "raw_image": raw_path,
                                "analysed_image": analysed_path
                            }
                            st.session_state.processed_state_key = current_state_key
                            st.success("Pose analysis completed successfully!")
                        except Exception as e:
                            st.error(f"Error processing image: {e}")
                            
        with col2:
            st.markdown("### Latest Capture & AI Analysis")
            
            if st.session_state.captured_image_info:
                info = st.session_state.captured_image_info
                
                # Layout for raw/analyzed photos
                pic_tab1, pic_tab2 = st.tabs(["✨ Final Photo (Clean)", "📊 AI Skeleton Analysis"])
                with pic_tab1:
                    st.image(info["raw_image"], width="stretch", caption=f"Captured Pose: {info['pose_name']}")
                with pic_tab2:
                    st.image(info["analysed_image"], width="stretch", caption="Overlay Skeleton Tracking")
                
                # Match stats
                st.markdown(f"#### Alignment Accuracy: **{info['score']}%**")
                
                # Progress bar colored by score
                if info["score"] >= 80:
                    st.progress(info["score"] / 100.0)
                    st.success("⭐ Excellent pose alignment! You look ready for the cover!")
                elif info["score"] >= 50:
                    st.progress(info["score"] / 100.0)
                    st.warning("👍 Good effort! Just a few minor adjustments needed.")
                else:
                    st.progress(info["score"] / 100.0)
                    st.error("📉 Needs work. Try practicing with the guides again!")
                
                # Joint feedback checklist
                st.markdown("##### **AI Feedback Details:**")
                for sugg in info["suggestions"]:
                    st.markdown(f"- 📢 {sugg}")
                
                # Download actions side-by-side
                st.markdown("---")
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    try:
                        with open(info["raw_image"], "rb") as f:
                            st.download_button(
                                label="💾 Download Clean Image",
                                data=f.read(),
                                file_name=f"pose_clean_{info['timestamp']}.png",
                                mime="image/png",
                                key=f"dl_latest_raw_{info['timestamp']}",
                                width="stretch"
                            )
                    except Exception:
                        st.error("Error loading clean photo for download.")
                with dl_col2:
                    try:
                        with open(info["analysed_image"], "rb") as f:
                            st.download_button(
                                label="📊 Download AI Tracking Overlay",
                                data=f.read(),
                                file_name=f"pose_tracked_{info['timestamp']}.png",
                                mime="image/png",
                                key=f"dl_latest_analysed_{info['timestamp']}",
                                width="stretch"
                            )
                    except Exception:
                        st.error("Error loading analyzed photo for download.")
            else:
                st.info("Capture a photo or upload an image to start analysis!")
                
    with tab_gallery:
        st.markdown("### Clicked Photos Gallery")
        
        # Scan captured photos
        raw_photos = sorted(glob.glob(os.path.join("assets", "captures", "raw_*.png")), reverse=True)
        
        if not raw_photos:
            st.info("No captured photos yet. Strike a pose and click one!")
        else:
            # Render grid of photos
            cols = st.columns(3)
            for idx, photo_path in enumerate(raw_photos):
                col = cols[idx % 3]
                filename = os.path.basename(photo_path)
                timestamp_str = filename.replace("raw_", "").replace(".png", "")
                try:
                    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(timestamp_str)))
                except ValueError:
                    formatted_time = "Unknown date"
                    
                with col:
                    st.image(photo_path, width="stretch", caption=f"Captured: {formatted_time}")
                    analysed_path = photo_path.replace("raw_", "analysed_")
                    
                    # Buttons container side-by-side
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        try:
                            with open(photo_path, "rb") as f:
                                st.download_button(
                                    label="💾 Download",
                                    data=f.read(),
                                    file_name=f"captured_pose_{timestamp_str}.png",
                                    mime="image/png",
                                    key=f"dl_gallery_{timestamp_str}",
                                    width="stretch"
                                )
                        except Exception:
                            st.error("Error")
                            
                    with btn_col2:
                        if st.button("🗑️ Delete", key=f"del_gallery_{timestamp_str}", width="stretch"):
                            try:
                                # Remove files
                                if os.path.exists(photo_path):
                                    os.remove(photo_path)
                                if os.path.exists(analysed_path):
                                    os.remove(analysed_path)
                                
                                # Clear latest capture in session state if it matches deleted image
                                if (st.session_state.captured_image_info and 
                                    str(st.session_state.captured_image_info["timestamp"]) == timestamp_str):
                                    st.session_state.captured_image_info = None
                                    st.session_state.processed_state_key = None
                                
                                st.success("Pose deleted!")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                                
                    # Option to view tracked overlay image
                    if os.path.exists(analysed_path):
                        if st.button("View AI Skeleton Overlay", key=f"overlay_{timestamp_str}", width="stretch"):
                            st.image(analysed_path, width="stretch", caption="Joint Tracking Overlay")

    with tab_create:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ➕ Record & Create a Custom Pose")
        st.markdown("Strike a custom pose, specify a name and description, and save it to your local library.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        pose_name_input = st.text_input("Pose Name (e.g. 'Warrior Pose')", key="custom_pose_name")
        pose_desc_input = st.text_area("Pose Description", key="custom_pose_desc")
        
        create_method = st.radio("Capture Method:", ["📷 Use Browser Camera", "📤 Upload Photo"], key="create_method")
        
        create_file = None
        if create_method == "📷 Use Browser Camera":
            create_file = st.camera_input("Strike your custom pose!", key="create_camera")
        else:
            create_file = st.file_uploader("Upload custom pose image", type=["png", "jpg", "jpeg"], key="create_uploader")
            
        if create_file is not None:
            file_bytes = create_file.getvalue()
            
            with st.spinner("Analyzing custom pose..."):
                try:
                    # Load image
                    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    detector = PoseDetector()
                    detector.process_frame(img_bgr)
                    lm_list = detector.find_landmarks(img_bgr, draw=False)
                    
                    if not lm_list:
                        st.error("No pose detected in the image! Please ensure your full body is visible in the frame.")
                    else:
                        # Draw detected skeleton for visual verification
                        img_preview = img_bgr.copy()
                        detector.find_landmarks(img_preview, draw=True, draw_connections=True, color=(0, 255, 204))
                        
                        st.image(img_preview, channels="BGR", width="stretch", caption="Detected Pose Skeleton")
                        
                        if not pose_name_input.strip():
                            st.warning("Please enter a name for this custom pose before saving.")
                        else:
                            if st.button("💾 Save Custom Pose to Library", type="primary", width="stretch"):
                                # Extract angles and landmarks
                                angles = get_pose_angles(lm_list)
                                landmarks_data = []
                                for item in lm_list:
                                    landmarks_data.append({
                                        "id": item[0],
                                        "x": item[3], # norm_x
                                        "y": item[4], # norm_y
                                        "vis": item[6] # visibility
                                    })
                                    
                                custom_poses = {}
                                json_path = os.path.join("assets", "poses", "custom_poses.json")
                                if os.path.exists(json_path):
                                    try:
                                        with open(json_path, 'r') as f:
                                            custom_poses = json.load(f)
                                    except Exception as e:
                                        st.error(f"Error loading existing poses: {e}")
                                        
                                custom_poses[pose_name_input.strip()] = {
                                    "description": pose_desc_input.strip() or f"Custom pose: {pose_name_input.strip()}",
                                    "angles": angles,
                                    "landmarks": landmarks_data,
                                    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                
                                try:
                                    with open(json_path, 'w') as f:
                                        json.dump(custom_poses, f, indent=4)
                                    st.success(f"Pose '{pose_name_input}' has been successfully added to your Pose Library!")
                                    time.sleep(1.0)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error saving pose: {e}")
                except Exception as e:
                    st.error(f"Error analyzing custom pose: {e}")

if __name__ == "__main__":
    main()
