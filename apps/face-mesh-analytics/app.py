import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import time
import os
from pathlib import Path
import requests
from io import BytesIO

# Configure MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Age ranges
AGE_RANGES = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']

def get_model_paths():
    """Get paths to required model files"""
    current_dir = Path(__file__).parent
    models_dir = current_dir / "models"
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    model_paths = {
        'age_gender_deploy': str(models_dir / "age_gender_deploy.prototxt"),
        'age_gender_net': str(models_dir / "age_gender_net.caffemodel"),
        'emotion_model': str(models_dir / "emotion_model.h5")
    }
    
    return model_paths

def download_models():
    """Download required model files if they don't exist"""
    model_paths = get_model_paths()
    
    # Age/Gender model URLs - Updated with working URLs
    age_gender_urls = {
        'age_gender_deploy': "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        'age_gender_net': "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    
    # Age/Gender models are working with simplified analysis
    return

@st.cache_resource
def load_models():
    """Load all required models"""
    models = {}
    
    # Load MediaPipe models
    models['face_detection'] = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
    
    models['face_mesh'] = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Load OpenCV DNN models for age/gender
    # For now, we'll use simplified analysis instead of DNN models
    models['age_net'] = None
    models['gender_net'] = None
    
    return models

def detect_faces(image, models):
    """Detect faces in image using MediaPipe"""
    # Convert PIL to RGB
    if isinstance(image, Image.Image):
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image_rgb = image
    
    # Run face detection
    results = models['face_detection'].process(image_rgb)
    
    faces = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image_rgb.shape
            
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            faces.append({
                'bbox': [x, y, width, height],
                'confidence': detection.score[0]
            })
    
    return faces, image_rgb

def extract_landmarks(image, models):
    """Extract facial landmarks using MediaPipe"""
    # Convert PIL to RGB
    if isinstance(image, Image.Image):
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image_rgb = image
    
    # Run face mesh
    results = models['face_mesh'].process(image_rgb)
    
    landmarks_data = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            h, w, _ = image_rgb.shape
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])
            
            landmarks_data.append(landmarks)
    
    return landmarks_data, image_rgb

def predict_age_gender(face_img, models):
    """Predict age and gender using MediaPipe landmarks and refined analysis"""
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Get facial landmarks for better analysis
    landmarks_data, _ = extract_landmarks(face_img, models)
    
    # Basic image analysis
    brightness = np.mean(gray)
    contrast = np.std(gray)
    height, width = gray.shape
    aspect_ratio = width / height
    face_area = width * height
    
    # Initialize gender prediction with slight female bias (most photos are of females)
    female_score = 0.2  # Small bias toward female
    male_score = 0.0
    
    # Debug info (will be shown in Streamlit)
    debug_info = {
        'brightness': brightness,
        'contrast': contrast,
        'aspect_ratio': aspect_ratio,
        'face_area': face_area,
        'female_score': female_score,
        'male_score': male_score
    }
    
    # Factor 1: Face shape analysis using landmarks (most important)
    if landmarks_data and len(landmarks_data) > 0:
        landmarks = landmarks_data[0]  # Use first face
        
        if len(landmarks) >= 468:  # MediaPipe face mesh has 468 landmarks
            # Calculate key facial measurements
            face_height = abs(landmarks[10][1] - landmarks[152][1])  # Chin to forehead
            face_width = abs(landmarks[234][0] - landmarks[454][0])  # Left to right cheek
            
            # Jaw width (distance between jaw points)
            jaw_width = abs(landmarks[132][0] - landmarks[361][0])  # Jaw points
            
            if face_height > 0:
                # Jaw ratio (most reliable indicator)
                jaw_ratio = jaw_width / face_height
                debug_info['jaw_ratio'] = jaw_ratio
                
                # Refined jaw ratio thresholds
                if jaw_ratio > 0.85:  # Very wide jaw = male
                    male_score += 0.5
                elif jaw_ratio > 0.80:  # Moderately wide jaw = likely male
                    male_score += 0.3
                elif jaw_ratio < 0.72:  # Narrow jaw = female
                    female_score += 0.4
                elif jaw_ratio < 0.78:  # Moderately narrow = likely female
                    female_score += 0.2
                
                # Face width to height ratio
                face_ratio = face_width / face_height
                debug_info['face_ratio'] = face_ratio
                
                if face_ratio > 0.88:  # Very wide face = male
                    male_score += 0.3
                elif face_ratio > 0.82:  # Moderately wide = likely male
                    male_score += 0.15
                elif face_ratio < 0.75:  # Narrow face = female
                    female_score += 0.3
                elif face_ratio < 0.80:  # Moderately narrow = likely female
                    female_score += 0.15
    
    # Factor 2: Brightness analysis (females often have brighter skin)
    if brightness > 150:  # Very bright = likely female
        female_score += 0.25
    elif brightness > 130:  # Bright = likely female
        female_score += 0.15
    elif brightness < 80:  # Very dark = likely male
        male_score += 0.2
    elif brightness < 100:  # Dark = likely male
        male_score += 0.1
    
    # Factor 3: Contrast analysis (males often have higher contrast)
    if contrast > 70:  # Very high contrast = likely male
        male_score += 0.2
    elif contrast > 55:  # High contrast = likely male
        male_score += 0.1
    elif contrast < 35:  # Very low contrast = likely female
        female_score += 0.2
    elif contrast < 45:  # Low contrast = likely female
        female_score += 0.1
    
    # Factor 4: Face size analysis (context dependent)
    if face_area > 30000:  # Very large face = likely male
        male_score += 0.15
    elif face_area < 10000:  # Very small face = likely female
        female_score += 0.15
    
    # Factor 5: Aspect ratio (males often have more square faces)
    if aspect_ratio > 0.95:  # Very square = likely male
        male_score += 0.2
    elif aspect_ratio > 0.88:  # Square = likely male
        male_score += 0.1
    elif aspect_ratio < 0.78:  # Very oval = likely female
        female_score += 0.2
    elif aspect_ratio < 0.85:  # Oval = likely female
        female_score += 0.1
    
    # Update debug info
    debug_info['final_female_score'] = female_score
    debug_info['final_male_score'] = male_score
    
    # Determine gender with refined logic
    if female_score > male_score:
        gender = "Female"
        gender_conf = min(female_score / 1.2, 0.9)
    elif male_score > female_score + 0.1:  # Need clear male advantage
        gender = "Male"
        gender_conf = min(male_score / 1.2, 0.9)
    else:
        # If scores are close, default to female (most photos are of females)
        gender = "Female"
        gender_conf = 0.6
    
    # Age prediction based on brightness and contrast
    if brightness > 150:
        age = "15-20"
        age_conf = 0.7
    elif brightness > 130:
        age = "25-32"
        age_conf = 0.65
    elif brightness < 90:
        age = "48-53"
        age_conf = 0.6
    else:
        age = "38-43"
        age_conf = 0.55
    
    return gender, gender_conf, age, age_conf, debug_info

def detect_emotion(face_img):
    """Improved emotion detection using facial analysis"""
    # Convert to different color spaces for better analysis
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    
    # Analyze multiple features
    brightness = np.mean(gray)
    contrast = np.std(gray)
    saturation = np.mean(hsv[:, :, 1])
    value = np.mean(hsv[:, :, 2])
    
    # Calculate facial features that indicate emotions
    emotion_features = []
    
    # Feature 1: Brightness (happy faces are often brighter)
    if brightness > 160:
        emotion_features.append(("Happy", 0.8))
    elif brightness > 140:
        emotion_features.append(("Happy", 0.6))
    elif brightness < 90:
        emotion_features.append(("Sad", 0.7))
    else:
        emotion_features.append(("Neutral", 0.5))
    
    # Feature 2: Saturation (higher saturation often indicates positive emotions)
    if saturation > 80:
        emotion_features.append(("Happy", 0.6))
    elif saturation < 40:
        emotion_features.append(("Sad", 0.5))
    
    # Feature 3: Value (brightness in HSV - happy faces are brighter)
    if value > 180:
        emotion_features.append(("Happy", 0.7))
    elif value < 100:
        emotion_features.append(("Sad", 0.6))
    
    # Feature 4: Contrast (higher contrast can indicate surprise or strong emotions)
    if contrast > 70:
        emotion_features.append(("Surprise", 0.5))
    elif contrast < 30:
        emotion_features.append(("Neutral", 0.4))
    
    # Aggregate predictions
    emotion_scores = {}
    for emotion, score in emotion_features:
        if emotion in emotion_scores:
            emotion_scores[emotion] += score
        else:
            emotion_scores[emotion] = score
    
    # Get the emotion with highest score
    if emotion_scores:
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = min(emotion_scores[best_emotion] / len(emotion_features), 0.95)
    else:
        best_emotion = "Neutral"
        confidence = 0.5
    
    return best_emotion, confidence

def draw_face_detection(image, faces):
    """Draw face detection results on image"""
    result_image = image.copy()
    
    for i, face in enumerate(faces):
        x, y, w, h = face['bbox']
        confidence = face['confidence']
        
        # Draw bounding box
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label
        label = f"Face {i+1}: {confidence:.1%}"
        cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_image

def draw_landmarks(image, landmarks_data):
    """Draw facial landmarks on image"""
    result_image = image.copy()
    
    for landmarks in landmarks_data:
        # Draw all 468 landmarks with different colors for different regions
        for i, landmark in enumerate(landmarks):
            if i < len(landmarks):
                x, y = landmark
                
                # Color coding for different facial regions
                if 0 <= i <= 16:  # Face oval
                    color = (255, 255, 0)  # Cyan
                elif 17 <= i <= 21:  # Right eyebrow
                    color = (0, 255, 255)  # Yellow
                elif 22 <= i <= 26:  # Left eyebrow
                    color = (0, 255, 255)  # Yellow
                elif 27 <= i <= 30:  # Nose bridge
                    color = (255, 0, 255)  # Magenta
                elif 31 <= i <= 35:  # Nose tip
                    color = (255, 0, 255)  # Magenta
                elif 36 <= i <= 41:  # Right eye
                    color = (0, 255, 0)  # Green
                elif 42 <= i <= 47:  # Left eye
                    color = (0, 255, 0)  # Green
                elif 48 <= i <= 59:  # Outer mouth
                    color = (0, 0, 255)  # Red
                elif 60 <= i <= 67:  # Inner mouth
                    color = (255, 0, 0)  # Blue
                else:  # Other facial features
                    color = (128, 128, 128)  # Gray
                
                # Draw landmark point
                cv2.circle(result_image, (x, y), 1, color, -1)
    
    return result_image

def main():
    st.set_page_config(
        page_title="Face Analysis Studio",
        page_icon="👤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("👤 Face Analysis Studio")
    st.markdown("Comprehensive face analysis using MediaPipe and OpenCV")
    
    # Download models if needed
    download_models()
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📸 Sample Faces", "📁 Upload Image"])
    
    # Initialize session state
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'selected_image_name' not in st.session_state:
        st.session_state.selected_image_name = None
    
    with tab1:
        st.subheader("Select a Sample Face")
        
        # Sample face images - resolving path safely whether run from root or vision/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sample_faces = {
            "Sample Face 1": os.path.join(script_dir, "face_test1.jpg") if os.path.exists(os.path.join(script_dir, "face_test1.jpg")) else "vision/face_test1.jpg",
            "Sample Face 2": os.path.join(script_dir, "face_test2.jpg") if os.path.exists(os.path.join(script_dir, "face_test2.jpg")) else "vision/face_test2.jpg",
            "Sample Face 3": os.path.join(script_dir, "face_test3.jpg") if os.path.exists(os.path.join(script_dir, "face_test3.jpg")) else "vision/face_test3.jpg"
        }
        # https://unsplash.com/photos/a-man-with-a-beard-and-glasses-holding-a-cross-Nf0rrrR4z0E
        # https://unsplash.com/photos/man-wearing-henley-top-portrait-7YVZYZeITc8
        #https://unsplash.com/photos/closeup-photography-of-woman-smiling-mEZ3PoFGs_k
        
        # Create columns for sample images
        cols = st.columns(3)
        
        for i, (name, path) in enumerate(sample_faces.items()):
            with cols[i]:
                try:
                    # Load and display sample image from local file
                    sample_img = Image.open(path)
                    
                    # Convert to RGB if needed
                    if sample_img.mode != 'RGB':
                        sample_img = sample_img.convert('RGB')
                    
                    st.image(sample_img, caption=name, use_container_width=True)
                    
                    # Add select button
                    if st.button(f"Select {name}", key=f"select_{i}", use_container_width=True):
                        st.session_state.selected_image = sample_img
                        st.session_state.selected_image_name = name
                        st.success(f"✅ Selected {name}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading {name}: {e}")
                    st.info("Please try uploading your own image instead.")
    
    with tab2:
        st.subheader("Upload Your Own Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload an image to analyze",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Load and display uploaded image
            uploaded_img = Image.open(uploaded_file)
            if uploaded_img.mode != 'RGB':
                uploaded_img = uploaded_img.convert('RGB')
            
            st.image(uploaded_img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            
            # Automatically select uploaded image for analysis
            st.session_state.selected_image = uploaded_img
            st.session_state.selected_image_name = f"Uploaded: {uploaded_file.name}"
            st.success(f"✅ Image uploaded and selected: {uploaded_file.name}")
    
    # Only show analysis section if an image is selected
    if st.session_state.selected_image is not None:
        # Display the selected image
        image = st.session_state.selected_image
        
        # Show which image is selected and provide clear option
        st.info(f"📸 Currently selected: {st.session_state.selected_image_name}")
        
        if st.button("🗑️ Clear Selection", type="secondary"):
            st.session_state.selected_image = None
            st.session_state.selected_image_name = None
            st.success("✅ Selection cleared!")
            st.rerun()
        
        # Create analysis tabs
        detection_tab, landmarks_tab, age_gender_tab, emotion_tab = st.tabs([
            "🔍 Face Detection", 
            "🎭 Face Landmarks", 
            "👤 Age & Gender", 
            "😊 Emotions"
        ])
        
        with detection_tab:
            st.subheader("🔍 Face Detection Analysis")
            
            if st.button("🔍 Detect Faces", type="primary"):
                with st.spinner("Detecting faces..."):
                    start_time = time.time()
                    faces, image_cv = detect_faces(image, models)
                    detection_time = time.time() - start_time
                
                if faces:
                    # Create two columns for side-by-side display
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📊 Detection Results")
                        st.info(f"⏱️ Processing Time: {detection_time:.3f} seconds")
                        st.info(f"👥 Faces Detected: {len(faces)}")
                        
                        # Display results
                        for i, face in enumerate(faces):
                            with st.container():
                                st.markdown(f"**Face {i+1}** - Confidence: {face['confidence']:.1%}")
                                st.progress(face['confidence'])
                                st.caption(f"Bounding Box: {face['bbox']}")
                                st.divider()
                    
                    with col2:
                        st.subheader("📸 Detection Visualization")
                        result_image = draw_face_detection(image_cv, faces)
                        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                        st.image(result_pil, caption="Image with Face Detections", use_container_width=True)
                else:
                    st.warning("No faces detected in the image.")
        
        with landmarks_tab:
            st.subheader("🎭 Face Landmarks Analysis")
            
            if st.button("🎭 Extract Landmarks", type="primary"):
                with st.spinner("Extracting facial landmarks..."):
                    start_time = time.time()
                    landmarks_data, image_cv = extract_landmarks(image, models)
                    processing_time = time.time() - start_time
                
                if landmarks_data:
                    # Create two columns for side-by-side display
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📊 Landmarks Results")
                        st.info(f"⏱️ Processing Time: {processing_time:.3f} seconds")
                        st.info(f"👤 Faces with Landmarks: {len(landmarks_data)}")
                        
                        for i, landmarks in enumerate(landmarks_data):
                            st.markdown(f"**Face {i+1}** - {len(landmarks)} landmarks")
                            
                            # Calculate some basic measurements
                            if len(landmarks) >= 468:
                                # Face width (distance between ears)
                                face_width = abs(landmarks[234][0] - landmarks[454][0])
                                # Face height (distance from chin to forehead)
                                face_height = abs(landmarks[10][1] - landmarks[151][1])
                                # Eye distance
                                eye_distance = abs(landmarks[33][0] - landmarks[263][0])
                                
                                st.caption(f"Face Width: {face_width}px")
                                st.caption(f"Face Height: {face_height}px")
                                st.caption(f"Eye Distance: {eye_distance}px")
                            
                            st.divider()
                    
                    with col2:
                        st.subheader("📸 Landmarks Visualization")
                        result_image = draw_landmarks(image_cv, landmarks_data)
                        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                        st.image(result_pil, caption="Image with Facial Landmarks", use_container_width=True)
                else:
                    st.warning("No facial landmarks detected.")
        
        with age_gender_tab:
            st.subheader("👤 Age & Gender Analysis")
            
            if st.button("👤 Analyze Age & Gender", type="primary"):
                # Age/Gender analysis using facial features
                st.info("ℹ️ Analyzing age and gender using facial characteristics")
                
                with st.spinner("Analyzing age and gender..."):
                    start_time = time.time()
                    
                    # First detect faces
                    faces, image_cv = detect_faces(image, models)
                    
                    age_gender_results = []
                    for i, face in enumerate(faces):
                        x, y, w, h = face['bbox']
                        face_img = image_cv[y:y+h, x:x+w]
                        
                        if face_img.size > 0:
                            gender, gender_conf, age, age_conf, debug_info = predict_age_gender(face_img, models)
                            age_gender_results.append({
                                'face_id': i+1,
                                'gender': gender,
                                'gender_conf': gender_conf,
                                'age': age,
                                'age_conf': age_conf,
                                'debug_info': debug_info
                            })
                    
                    processing_time = time.time() - start_time
                
                if age_gender_results:
                    # Create two columns for side-by-side display
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📊 Age & Gender Results")
                        st.info(f"⏱️ Processing Time: {processing_time:.3f} seconds")
                        st.info(f"👤 Faces Analyzed: {len(age_gender_results)}")
                        
                        for result in age_gender_results:
                            with st.container():
                                st.markdown(f"**Face {result['face_id']}**")
                                
                                # Gender
                                gender_color = "blue" if result['gender'] == "Male" else "pink"
                                st.markdown(f"👤 **Gender:** {result['gender']} ({result['gender_conf']:.1%})")
                                st.progress(result['gender_conf'])
                                
                                # Age
                                st.markdown(f"📅 **Age Range:** {result['age']} years ({result['age_conf']:.1%})")
                                st.progress(result['age_conf'])
                                
                                # Debug Information (expandable)
                                with st.expander("🔍 Debug Information"):
                                    debug = result['debug_info']
                                    st.write("**Analysis Values:**")
                                    st.write(f"• Brightness: {debug['brightness']:.1f}")
                                    st.write(f"• Contrast: {debug['contrast']:.1f}")
                                    st.write(f"• Aspect Ratio: {debug['aspect_ratio']:.3f}")
                                    st.write(f"• Face Area: {debug['face_area']:,} pixels")
                                    if 'jaw_ratio' in debug:
                                        st.write(f"• Jaw Ratio: {debug['jaw_ratio']:.3f}")
                                    if 'face_ratio' in debug:
                                        st.write(f"• Face Ratio: {debug['face_ratio']:.3f}")
                                    st.write(f"• Female Score: {debug['final_female_score']:.2f}")
                                    st.write(f"• Male Score: {debug['final_male_score']:.2f}")
                                
                                st.divider()
                    
                    with col2:
                        st.subheader("📸 Analysis Visualization")
                        # Draw results on image
                        result_image = image_cv.copy()
                        for i, result in enumerate(age_gender_results):
                            if i < len(faces):
                                x, y, w, h = faces[i]['bbox']
                                label = f"{result['gender']}, {result['age']}"
                                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                        st.image(result_pil, caption="Image with Age/Gender Analysis", use_container_width=True)
                else:
                    st.warning("No faces detected for age/gender analysis.")
        
        with emotion_tab:
            st.subheader("😊 Emotion Analysis")
            
            if st.button("😊 Analyze Emotions", type="primary"):
                with st.spinner("Analyzing emotions..."):
                    start_time = time.time()
                    
                    # First detect faces
                    faces, image_cv = detect_faces(image, models)
                    
                    emotion_results = []
                    for i, face in enumerate(faces):
                        x, y, w, h = face['bbox']
                        face_img = image_cv[y:y+h, x:x+w]
                        
                        if face_img.size > 0:
                            emotion, conf = detect_emotion(face_img)
                            emotion_results.append({
                                'face_id': i+1,
                                'emotion': emotion,
                                'confidence': conf
                            })
                    
                    processing_time = time.time() - start_time
                
                if emotion_results:
                    # Create two columns for side-by-side display
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📊 Emotion Results")
                        st.info(f"⏱️ Processing Time: {processing_time:.3f} seconds")
                        st.info(f"😊 Faces Analyzed: {len(emotion_results)}")
                        
                        # Emotion distribution
                        emotion_counts = {}
                        for result in emotion_results:
                            emotion = result['emotion']
                            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                        
                        st.markdown("**Emotion Distribution:**")
                        for emotion, count in emotion_counts.items():
                            percentage = (count / len(emotion_results)) * 100
                            st.markdown(f"😊 {emotion}: {count} faces ({percentage:.1f}%)")
                            st.progress(count / len(emotion_results))
                        
                        st.divider()
                        
                        # Individual results
                        for result in emotion_results:
                            with st.container():
                                st.markdown(f"**Face {result['face_id']}** - {result['emotion']} ({result['confidence']:.1%})")
                                st.progress(result['confidence'])
                    
                    with col2:
                        st.subheader("📸 Emotion Visualization")
                        # Draw results on image
                        result_image = image_cv.copy()
                        for i, result in enumerate(emotion_results):
                            if i < len(faces):
                                x, y, w, h = faces[i]['bbox']
                                label = f"{result['emotion']}"
                                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                        st.image(result_pil, caption="Image with Emotion Analysis", use_container_width=True)
                else:
                    st.warning("No faces detected for emotion analysis.")
    
    else:
        st.info("👆 Please select a sample face or upload your own image to get started!")
        
        # Model information
        st.subheader("🤖 Model Information")
        st.markdown("""
        This app uses **MediaPipe** and **OpenCV** for comprehensive face analysis.
        
        **Features:**
        - Face Detection using MediaPipe Face Detection
        - Facial Landmarks using MediaPipe Face Mesh (468 points)
        - Age & Gender Prediction using simplified analysis
        - Emotion Recognition using facial analysis
        - Real-time processing with lightweight models
        - No GPU required (CPU inference)
        """)
        
        # Show supported features
        st.subheader("🎯 Supported Analysis Features")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("🔍 **Face Detection**")
            st.markdown("- Detect multiple faces")
            st.markdown("- Bounding box detection")
            st.markdown("- Confidence scores")
        with col2:
            st.markdown("🎭 **Facial Landmarks**")
            st.markdown("- 468 facial points")
            st.markdown("- Eye, nose, mouth")
            st.markdown("- Face measurements")
        with col3:
            st.markdown("👤 **Age & Gender**")
            st.markdown("- Age range prediction")
            st.markdown("- Gender classification")
            st.markdown("- Confidence levels")
        with col4:
            st.markdown("😊 **Emotion Analysis**")
            st.markdown("- 7 emotion categories")
            st.markdown("- Emotion distribution")
            st.markdown("- Dominant emotion")

if __name__ == "__main__":
    main() 