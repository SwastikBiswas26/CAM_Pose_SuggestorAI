import cv2
import numpy as np
import streamlit as st
import os
import time
import glob
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

def run_camera_feed(pose_name, pose_data):
    """
    Launches the live OpenCV camera loop, overlaying guidelines and matching in real-time.
    """
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open your webcam. Please check connection and permissions.")
        return None

    target_angles = pose_data["angles"]
    landmarks_template = pose_data.get("landmarks", [])
    
    status_placeholder = st.empty()
    status_placeholder.warning("🎥 Camera window is active! Position yourself in the frame. Press SPACEBAR to take a picture, or Q to exit.")
    
    captured_data = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Horizontal Flip for intuitive mirror view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # 2. Draw Target silhouette overlay (mirrored x coordinate)
        if landmarks_template:
            mirrored_template = []
            for lm in landmarks_template:
                mirrored_template.append({
                    "id": lm["id"],
                    "x": 1.0 - lm["x"],  # Mirror the template x coordinate to match mirrored camera feed
                    "y": lm["y"]
                })
            detector.draw_guide_silhouette(frame, mirrored_template, color=(150, 150, 150), thickness=2)
            
        # 3. Detect user's pose
        detector.process_frame(frame)
        
        # 4. Analyze and Draw live skeleton based on matching score
        lm_list = detector.find_landmarks(frame, draw=False)
        
        score = 0
        suggestions = ["Position yourself fully in the camera view"]
        status_dict = {}
        skeleton_color = (0, 0, 255) # Red by default (needs adjustment)
        
        if lm_list:
            user_angles = get_pose_angles(lm_list)
            score, suggestions, status_dict = get_feedback(user_angles, target_angles)
            
            # Determine skeleton color based on score
            if score >= 80:
                skeleton_color = (0, 255, 0) # Green (Perfect)
            elif score >= 50:
                skeleton_color = (0, 255, 255) # Yellow (Close)
            else:
                skeleton_color = (0, 0, 255) # Red (Needs work)
                
            # Draw skeleton with match-dependent color
            detector.find_landmarks(frame, draw=True, draw_connections=True, color=skeleton_color)
            
            # Draw Match score on screen
            cv2.putText(frame, f"Match: {score}%", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, skeleton_color, 2, cv2.LINE_AA)
            
            # Draw top feedback suggestion on screen
            if suggestions:
                cv2.putText(frame, suggestions[0], (10, h - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Pose Detected", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            
        # Display the live feed window
        cv2.imshow(f"AI Pose Assistant - Target: {pose_name}", frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27: # Q or Esc
            break
        elif key == 32: # Spacebar to capture
            # Save raw picture (without lines) and analyzed picture (with lines)
            timestamp = int(time.time())
            
            # Re-read a clean un-analyzed frame for the raw capture
            ret_raw, clean_frame = cap.read()
            if ret_raw:
                clean_frame = cv2.flip(clean_frame, 1)
            else:
                clean_frame = frame.copy() # fallback
                
            raw_path = f"assets/captures/raw_{timestamp}.png"
            analysed_path = f"assets/captures/analysed_{timestamp}.png"
            
            cv2.imwrite(raw_path, clean_frame)
            cv2.imwrite(analysed_path, frame)
            
            captured_data = {
                "timestamp": timestamp,
                "pose_name": pose_name,
                "score": score,
                "suggestions": suggestions,
                "status_dict": status_dict,
                "raw_image": raw_path,
                "analysed_image": analysed_path
            }
            break
            
    cap.release()
    cv2.destroyAllWindows()
    status_placeholder.empty()
    
    return captured_data

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
    st.sidebar.image(preview_img, channels="BGR", use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Tip: Stand far enough so your full body (head to ankles) is visible in the camera frame.")
    
    # Tab navigation
    tab_guide, tab_gallery = st.tabs(["🎯 Pose Guide & Camera", "🖼️ My Photo Gallery"])
    
    with tab_guide:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### How it works:")
            st.markdown("""
                1. Click **'Start AI Camera'** below.
                2. A live camera feed window will open.
                3. Align your joints with the **gray target guide lines** shown on screen.
                4. The camera overlay will show your score:
                   - 🔴 **Red Skeleton**: Keep adjusting!
                   - 🟡 **Yellow Skeleton**: Getting close!
                   - 🟢 **Green Skeleton**: Perfect alignment!
                5. Watch the feedback prompt at the bottom of the window.
                6. Press **SPACEBAR** to take the photo when ready!
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Start camera button
            if st.button("🚀 Start AI Camera", use_container_width=True, type="primary"):
                with st.spinner("Initializing webcam..."):
                    captured_info = run_camera_feed(selected_pose_name, pose_data)
                    if captured_info:
                        st.session_state.captured_image_info = captured_info
                        st.success("Photo clicked successfully! See results on the right.")
                        
        with col2:
            st.markdown("### Latest Capture & AI Analysis")
            
            if st.session_state.captured_image_info:
                info = st.session_state.captured_image_info
                
                # Layout for raw/analyzed photos
                pic_tab1, pic_tab2 = st.tabs(["✨ Final Photo (Clean)", "📊 AI skeleton Analysis"])
                with pic_tab1:
                    st.image(info["raw_image"], use_container_width=True, caption=f"Captured Pose: {info['pose_name']}")
                with pic_tab2:
                    st.image(info["analysed_image"], use_container_width=True, caption="Overlay Skeleton Tracking")
                
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
                                use_container_width=True
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
                                use_container_width=True
                            )
                    except Exception:
                        st.error("Error loading analyzed photo for download.")
            else:
                st.info("Click 'Start AI Camera' to take your first photo!")
                
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
                    st.image(photo_path, use_container_width=True, caption=f"Captured: {formatted_time}")
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
                                    use_container_width=True
                                )
                        except Exception:
                            st.error("Error")
                            
                    with btn_col2:
                        if st.button("🗑️ Delete", key=f"del_gallery_{timestamp_str}", use_container_width=True):
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
                                
                                st.success("Pose deleted!")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                                
                    # Option to view tracked overlay image
                    if os.path.exists(analysed_path):
                        if st.button("View AI Skeleton Overlay", key=f"overlay_{timestamp_str}", use_container_width=True):
                            st.image(analysed_path, use_container_width=True, caption="Joint Tracking Overlay")

if __name__ == "__main__":
    main()
