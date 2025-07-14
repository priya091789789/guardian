# streamlit_app.py - Streamlit UI Application
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import pickle
from PIL import Image
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GuardianMonitoringSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.svm_model = None
        self.label_encoder = None
        self.model_path = "models/guardian_model.pkl"
        self.recognizer_path = "models/face_recognizer.yml"
        self.encodings_path = "models/face_encodings.pkl"
        self.is_loaded = False

    def load_models(self):
        """Load trained models"""
        try:
            if not os.path.exists(self.model_path):
                st.error(f"❌ Model file not found: {self.model_path}")
                st.error("Please run the training script first: python train_model.py")
                return False

            if not os.path.exists(self.recognizer_path):
                st.error(f"❌ Recognizer file not found: {self.recognizer_path}")
                st.error("Please run the training script first: python train_model.py")
                return False

            self.face_recognizer.read(self.recognizer_path)

            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.svm_model = model_data['svm_model']
            self.label_encoder = model_data['label_encoder']
            self.training_info = {
                'training_date': model_data.get('training_date', 'Unknown'),
                'total_samples': model_data.get('total_samples', 0),
                'unique_persons': model_data.get('unique_persons', 0)
            }

            self.is_loaded = True
            logger.info("Models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            st.error(f"❌ Error loading models: {str(e)}")
            return False

    def preprocess_image(self, image):
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)

            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            gray = cv2.equalizeHist(gray)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            return gray, faces

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None, []

    def extract_features(self, face):
        try:
            face = cv2.resize(face, (200, 200))
            return face.flatten()
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None

    def recognize_face(self, image):
        try:
            if not self.is_loaded:
                return []

            gray, faces = self.preprocess_image(image)

            if gray is None or len(faces) == 0:
                return []

            recognized_faces = []

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (200, 200))

                label, confidence = self.face_recognizer.predict(face_resized)
                lbph_similarity = max(0, 1 - (confidence / 100))

                features = self.extract_features(face)
                if features is not None:
                    try:
                        features = features.reshape(1, -1)
                        expected_length = self.svm_model.n_features_in_
                        if len(features[0]) != expected_length:
                            features = features[:, :expected_length]

                        svm_prediction = self.svm_model.predict(features)[0]
                        svm_confidence = max(self.svm_model.predict_proba(features)[0])

                        if label == svm_prediction:
                            final_confidence = (lbph_similarity + svm_confidence) / 2
                            predicted_label = label
                        else:
                            if lbph_similarity > svm_confidence:
                                predicted_label = label
                                final_confidence = lbph_similarity * 0.8
                            else:
                                predicted_label = svm_prediction
                                final_confidence = svm_confidence * 0.8

                    except Exception as e:
                        logger.warning(f"SVM prediction failed: {str(e)}")
                        predicted_label = label
                        final_confidence = lbph_similarity * 0.7
                else:
                    predicted_label = label
                    final_confidence = lbph_similarity * 0.7

                try:
                    name = self.label_encoder.inverse_transform([predicted_label])[0]
                except:
                    name = f"Unknown_{predicted_label}"

                recognized_faces.append({
                    'name': name,
                    'confidence': final_confidence,
                    'location': (x, y, w, h),
                    'lbph_confidence': lbph_similarity,
                    'raw_lbph_confidence': confidence
                })

            return recognized_faces

        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            st.error(f"❌ Error in face recognition: {str(e)}")
            return []

    def draw_face_rectangles(self, image, face_results):
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)

            image_copy = image.copy()

            for result in face_results:
                x, y, w, h = result['location']

                if result['confidence'] >= 0.6:
                    color = (0, 255, 0)
                elif result['confidence'] >= 0.4:
                    color = (255, 165, 0)
                else:
                    color = (0, 0, 255)

                label = f"{result['name']}"
                confidence_text = f"{result['confidence']:.2%}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                cv2.rectangle(image_copy, (x, y - text_height - 10),
                              (x + text_width, y), color, -1)

                cv2.putText(image_copy, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
                cv2.putText(image_copy, confidence_text, (x, y + h + 20),
                            font, font_scale, color, thickness)

            return image_copy

        except Exception as e:
            logger.error(f"Error drawing rectangles: {str(e)}")
            return image

    def log_access_attempt(self, name, confidence, status):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} - {name} - Confidence: {confidence:.2%} - Status: {status}"

            with open("access_log.txt", "a") as f:
                f.write(log_entry + "\n")

            logger.info(log_entry)

        except Exception as e:
            logger.error(f"Error logging access attempt: {str(e)}")


# Complete main() function for Guardian Monitoring System

def main():
    st.set_page_config(
        page_title="Guardian Monitoring System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5em;
        margin-bottom: 30px;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
    </style>
    """, unsafe_allow_html=True)

    if 'guardian_system' not in st.session_state:
        st.session_state.guardian_system = GuardianMonitoringSystem()

    guardian_system = st.session_state.guardian_system

    st.markdown('<h1 class="main-header">🛡️ Guardian Monitoring System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    if not guardian_system.is_loaded:
        with st.spinner("Loading trained models..."):
            success = guardian_system.load_models()
            if success:
                st.success("✅ Models loaded successfully!")
            else:
                st.error("❌ Failed to load models. Please run training first.")
                st.stop()

    st.sidebar.title("📊 System Information")
    st.sidebar.markdown("### Model Status")
    st.sidebar.success("✅ Models Loaded")

    if hasattr(guardian_system, 'training_info'):
        st.sidebar.markdown("### Training Information")
        st.sidebar.info(f"**Training Date:** {guardian_system.training_info['training_date']}")
        st.sidebar.info(f"**Total Samples:** {guardian_system.training_info['total_samples']}")
        st.sidebar.info(f"**Unique Persons:** {guardian_system.training_info['unique_persons']}")

    st.sidebar.markdown("---")
    st.sidebar.title("🔧 Navigation")
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Live Camera Monitoring", "Upload Image Test", "System Logs"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Minimum confidence required for access approval"
    )

    if mode == "Live Camera Monitoring":
        st.header("📹 Live Camera Monitoring")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📸 Camera Input")
            camera_input = st.camera_input("Take a picture for guardian verification")

            if camera_input is not None:
                image = Image.open(camera_input)
                st.image(image, caption="Captured Image", use_container_width=True)

        with col2:
            st.subheader("🔍 Recognition Results")

            if camera_input is not None:
                with st.spinner("Analyzing faces..."):
                    recognized_faces = guardian_system.recognize_face(image)

                if recognized_faces:
                    for face in recognized_faces:
                        name = face['name']
                        confidence = face['confidence']

                        if confidence >= confidence_threshold:
                            st.markdown(f'''
                            <div class="status-success">
                                <h3>✅ ACCESS GRANTED</h3>
                                <p><strong>Guardian:</strong> {name}</p>
                                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                                <p><strong>Status:</strong> Verified Guardian</p>
                            </div>
                            ''', unsafe_allow_html=True)
                            guardian_system.log_access_attempt(name, confidence, "GRANTED")

                        elif confidence >= 0.4:
                            st.markdown(f'''
                            <div class="status-warning">
                                <h3>⚠️ LOW CONFIDENCE</h3>
                                <p><strong>Detected:</strong> {name}</p>
                                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                                <p><strong>Status:</strong> Please try again with better lighting</p>
                            </div>
                            ''', unsafe_allow_html=True)
                            guardian_system.log_access_attempt(name, confidence, "LOW_CONFIDENCE")

                        else:
                            st.markdown(f'''
                            <div class="status-error">
                                <h3>❌ ACCESS DENIED</h3>
                                <p><strong>Detection:</strong> {name}</p>
                                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                                <p><strong>Status:</strong> Not a registered guardian</p>
                            </div>
                            ''', unsafe_allow_html=True)
                            guardian_system.log_access_attempt(name, confidence, "DENIED")

                        with st.expander("🔍 Detailed Analysis"):
                            st.write(f"**LBPH Confidence:** {face['lbph_confidence']:.2%}")
                            st.write(f"**Raw LBPH Score:** {face['raw_lbph_confidence']:.2f}")
                            st.write(f"**Final Confidence:** {confidence:.2%}")

                    st.subheader("🎯 Face Detection Results")
                    annotated_image = guardian_system.draw_face_rectangles(image, recognized_faces)
                    st.image(annotated_image, caption="Annotated Image", use_container_width=True)

                else:
                    st.markdown(f'''
                    <div class="status-error">
                        <h3>❌ NO FACE DETECTED</h3>
                        <p><strong>Status:</strong> No face found in the image</p>
                        <p><strong>Action:</strong> Please ensure face is clearly visible</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    guardian_system.log_access_attempt("Unknown", 0.0, "NO_FACE_DETECTED")

    elif mode == "Upload Image Test":
        st.header("🖼️ Upload Image Test")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📁 Upload Image")
            uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'bmp'])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.markdown("### 📊 Image Information")
                st.write(f"**File name:** {uploaded_file.name}")
                st.write(f"**File size:** {uploaded_file.size} bytes")
                st.write(f"**Image size:** {image.size}")
                st.write(f"**Image mode:** {image.mode}")

        with col2:
            st.subheader("🔍 Recognition Results")

            if uploaded_file is not None:
                with st.spinner("Processing uploaded image..."):
                    recognized_faces = guardian_system.recognize_face(image)

                if recognized_faces:
                    st.success(f"✅ {len(recognized_faces)} face(s) detected")
                    for i, face in enumerate(recognized_faces):
                        name = face['name']
                        confidence = face['confidence']
                        st.markdown(f"### Face {i+1}")

                        if confidence >= confidence_threshold:
                            st.success(f"✅ **Guardian Recognized:** {name}")
                            st.info(f"🎯 **Confidence:** {confidence:.2%}")
                            st.info("🔓 **Status:** Access would be granted")
                            guardian_system.log_access_attempt(name, confidence, "GRANTED")
                        elif confidence >= 0.4:
                            st.warning(f"⚠️ **Possible Match:** {name}")
                            st.warning(f"🎯 **Confidence:** {confidence:.2%}")
                            st.warning("🔒 **Status:** Low confidence - access denied")
                            guardian_system.log_access_attempt(name, confidence, "LOW_CONFIDENCE")
                        else:
                            st.error(f"❌ **Not Recognized:** {name}")
                            st.error(f"🎯 **Confidence:** {confidence:.2%}")
                            st.error("🔒 **Status:** Access denied")
                            guardian_system.log_access_attempt(name, confidence, "DENIED")

                        with st.expander(f"🔍 Detailed Analysis - Face {i+1}"):
                            st.write(f"**LBPH Confidence:** {face['lbph_confidence']:.2%}")
                            st.write(f"**Raw LBPH Score:** {face['raw_lbph_confidence']:.2f}")
                            st.write(f"**Face Location:** {face['location']}")
                            st.write(f"**Final Confidence:** {confidence:.2%}")
                        st.markdown("---")

                    st.subheader("🎯 Annotated Image")
                    annotated_image = guardian_system.draw_face_rectangles(image, recognized_faces)
                    st.image(annotated_image, caption="Face Detection Results", use_container_width=True)

                else:
                    st.error("❌ No faces detected in the uploaded image")
                    st.info("💡 **Tips for better detection:**")
                    st.info("• Ensure the face is clearly visible")
                    st.info("• Use good lighting conditions")
                    st.info("• Face should be facing forward")
                    st.info("• Remove any obstructions (masks, sunglasses)")

    elif mode == "System Logs":
        st.header("📋 System Logs")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🔍 Access Logs")
            try:
                if os.path.exists("access_log.txt"):
                    with open("access_log.txt", "r") as f:
                        logs = f.readlines()

                    if logs:
                        st.subheader("📊 Recent Access Attempts")
                        recent_logs = logs[-20:][::-1]

                        for log in recent_logs:
                            parts = log.strip().split(" - ")
                            if len(parts) >= 4:
                                timestamp, name, confidence, status = parts[:4]
                                if "GRANTED" in status:
                                    st.success(f"✅ {timestamp} | {name} | {confidence} | {status}")
                                elif "DENIED" in status:
                                    st.error(f"❌ {timestamp} | {name} | {confidence} | {status}")
                                elif "LOW_CONFIDENCE" in status:
                                    st.warning(f"⚠️ {timestamp} | {name} | {confidence} | {status}")
                                else:
                                    st.info(f"ℹ️ {timestamp} | {name} | {confidence} | {status}")

                        st.download_button("📥 Download Complete Logs", "\n".join(logs),
                            file_name=f"access_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain")
                    else:
                        st.info("No access logs found")
                else:
                    st.info("No access logs file found")
            except Exception as e:
                st.error(f"Error reading logs: {str(e)}")

        with col2:
            st.subheader("📊 Statistics")
            try:
                if os.path.exists("access_log.txt"):
                    with open("access_log.txt", "r") as f:
                        logs = f.readlines()
                    if logs:
                        total = len(logs)
                        granted = sum(1 for l in logs if "GRANTED" in l)
                        denied = sum(1 for l in logs if "DENIED" in l)
                        low = sum(1 for l in logs if "LOW_CONFIDENCE" in l)
                        noface = sum(1 for l in logs if "NO_FACE_DETECTED" in l)

                        st.metric("Total Attempts", total)
                        st.metric("Access Granted", granted)
                        st.metric("Access Denied", denied)
                        st.metric("Low Confidence", low)
                        st.metric("No Face Detected", noface)

                        if total:
                            rate = granted / total * 100
                            st.metric("Success Rate", f"{rate:.1f}%")

                        if st.button("🗑️ Clear Logs"):
                            os.remove("access_log.txt")
                            st.success("Logs cleared successfully!")
                            st.rerun()
            except Exception as e:
                st.error(f"Error calculating statistics: {str(e)}")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🌟 Good Lighting**\nEnsure adequate lighting on the face")
    with col2:
        st.info("**📐 Proper Angle**\nFace should be looking directly at camera")
    with col3:
        st.info("**🔍 Clear View**\nRemove masks, sunglasses, or other obstructions")

    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.9em;">🛡️ Guardian Monitoring System - Ensuring Child Safety with Advanced Face Recognition</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
