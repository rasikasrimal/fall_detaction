import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import time
from collections import deque
import logging
import threading

# Try to import audio libraries
try:
    import pygame

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: pygame not available. Audio alarms disabled.")

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow as tf

        tflite = tf.lite
    except ImportError:
        print("Error: Neither tflite_runtime nor tensorflow is available")
        exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration settings for fall detection."""
    model_path: str = 'fall_transformer.tflite'
    input_timesteps: int = 30
    fall_threshold: float = 0.97
    min_keypoint_confidence: float = 0.3
    fall_cooldown: float = 10.0
    pose_complexity: int = 0
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    min_consecutive_falls: int = 5  # Added missing attribute
    alarm_duration: float = 3.0  # Added missing attribute


@dataclass
class KeypointData:
    """Keypoint mapping and indices."""
    # MediaPipe to custom keypoint mapping
    mp_to_custom = {
        mp.solutions.pose.PoseLandmark.NOSE: 'Nose',
        mp.solutions.pose.PoseLandmark.LEFT_EYE: 'Left Eye',
        mp.solutions.pose.PoseLandmark.RIGHT_EYE: 'Right Eye',
        mp.solutions.pose.PoseLandmark.LEFT_EAR: 'Left Ear',
        mp.solutions.pose.PoseLandmark.RIGHT_EAR: 'Right Ear',
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER: 'Left Shoulder',
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER: 'Right Shoulder',
        mp.solutions.pose.PoseLandmark.LEFT_ELBOW: 'Left Elbow',
        mp.solutions.pose.PoseLandmark.RIGHT_ELBOW: 'Right Elbow',
        mp.solutions.pose.PoseLandmark.LEFT_WRIST: 'Left Wrist',
        mp.solutions.pose.PoseLandmark.RIGHT_WRIST: 'Right Wrist',
        mp.solutions.pose.PoseLandmark.LEFT_HIP: 'Left Hip',
        mp.solutions.pose.PoseLandmark.RIGHT_HIP: 'Right Hip',
        mp.solutions.pose.PoseLandmark.LEFT_KNEE: 'Left Knee',
        mp.solutions.pose.PoseLandmark.RIGHT_KNEE: 'Right Knee',
        mp.solutions.pose.PoseLandmark.LEFT_ANKLE: 'Left Ankle',
        mp.solutions.pose.PoseLandmark.RIGHT_ANKLE: 'Right Ankle'
    }

    names = [
        'Left Ankle', 'Left Ear', 'Left Elbow', 'Left Eye', 'Left Hip',
        'Left Knee', 'Left Shoulder', 'Left Wrist', 'Nose', 'Right Ankle',
        'Right Ear', 'Right Elbow', 'Right Eye', 'Right Hip', 'Right Knee',
        'Right Shoulder', 'Right Wrist'
    ]

    name_to_idx = {name: i for i, name in enumerate(names)}
    num_features = len(names) * 3


class AudioManager:
    """Handles audio alarm functionality."""

    def __init__(self, config: Config):
        self.config = config
        self.alarm_active = False
        self.alarm_thread = None
        self.stop_alarm_flag = False

        if AUDIO_AVAILABLE:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

    def generate_alarm_sound(self, duration: float = 0.5, frequency: int = 1000):
        """Generate a simple alarm beep sound."""
        if not AUDIO_AVAILABLE:
            return None

        sample_rate = 22050
        frames = int(duration * sample_rate)
        arr = np.zeros((frames, 2), dtype=np.int16)

        # Generate sine wave
        for i in range(frames):
            wave = int(16383 * np.sin(2 * np.pi * frequency * i / sample_rate))
            arr[i] = [wave, wave]

        return pygame.sndarray.make_sound(arr)

    def start_alarm(self):
        """Start the alarm in a separate thread."""
        if not AUDIO_AVAILABLE or self.alarm_active:
            return

        self.alarm_active = True
        self.stop_alarm_flag = False
        self.alarm_thread = threading.Thread(target=self._alarm_loop)
        self.alarm_thread.daemon = True
        self.alarm_thread.start()

    def stop_alarm(self):
        """Stop the alarm."""
        self.stop_alarm_flag = True
        self.alarm_active = False

    def _alarm_loop(self):
        """Main alarm loop (runs in separate thread)."""
        if not AUDIO_AVAILABLE:
            return

        alarm_sound = self.generate_alarm_sound(0.5, 800)
        if alarm_sound is None:
            return

        start_time = time.time()

        while (time.time() - start_time < self.config.alarm_duration and
               not self.stop_alarm_flag):
            alarm_sound.play()
            time.sleep(0.6)  # Short gap between beeps

        self.alarm_active = False


class ModelManager:
    """Handles TFLite model loading and inference."""

    def __init__(self, config: Config):
        self.config = config
        self.interpreter: Optional[tflite.Interpreter] = None
        self.input_details: Optional[List[Dict[str, Any]]] = None
        self.output_details: Optional[List[Dict[str, Any]]] = None

    def load_model(self) -> bool:
        """Load and initialize TFLite model."""
        try:
            if not Path(self.config.model_path).exists():
                logger.error(f"Model file not found: {self.config.model_path}")
                return False

            self.interpreter = tflite.Interpreter(model_path=self.config.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Validate model shape
            expected_shape = tuple(self.input_details[0]['shape'])
            required_shape = (1, self.config.input_timesteps, KeypointData.num_features)

            if expected_shape != required_shape:
                logger.error(f"Model shape mismatch. Expected: {required_shape}, Got: {expected_shape}")
                return False

            logger.info(f"Model loaded successfully: {self.config.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, sequence: np.ndarray) -> float:
        """Run inference on feature sequence."""
        if self.interpreter is None:
            raise RuntimeError("Model not loaded")

        try:
            # Prepare input
            model_input = np.expand_dims(sequence, axis=0).astype(np.float32)

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], model_input)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            return float(output[0][0])

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0


class FeatureExtractor:
    """Handles pose landmark extraction and normalization."""

    def __init__(self, config: Config):
        self.config = config
        self.keypoint_data = KeypointData()

    def get_keypoint_indices(self, name: str) -> Tuple[int, int, int]:
        """Get x, y, confidence indices for a keypoint."""
        if name not in self.keypoint_data.name_to_idx:
            raise ValueError(f"Unknown keypoint: {name}")
        idx = self.keypoint_data.name_to_idx[name]
        return idx * 3, idx * 3 + 1, idx * 3 + 2

    def extract_features(self, pose_results) -> np.ndarray:
        """Extract and normalize pose features from MediaPipe results."""
        features = np.zeros(KeypointData.num_features, dtype=np.float32)

        if not pose_results.pose_landmarks:
            return features

        landmarks = pose_results.pose_landmarks.landmark

        # Extract raw coordinates
        for mp_landmark, custom_name in self.keypoint_data.mp_to_custom.items():
            if custom_name in self.keypoint_data.name_to_idx:
                try:
                    lm = landmarks[mp_landmark.value]
                    x_idx, y_idx, c_idx = self.get_keypoint_indices(custom_name)
                    features[x_idx] = lm.x
                    features[y_idx] = lm.y
                    features[c_idx] = lm.visibility
                except (IndexError, KeyError):
                    continue

        # Normalize features
        return self._normalize_features(features)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize skeleton features using torso reference points."""
        normalized = features.copy()

        # Get reference points
        ref_points = {
            'ls': 'Left Shoulder', 'rs': 'Right Shoulder',
            'lh': 'Left Hip', 'rh': 'Right Hip'
        }

        try:
            # Extract reference coordinates
            coords = {}
            for key, name in ref_points.items():
                x_idx, y_idx, c_idx = self.get_keypoint_indices(name)
                coords[key] = {
                    'x': features[x_idx], 'y': features[y_idx], 'c': features[c_idx]
                }

            # Calculate midpoints
            mid_shoulder = self._calculate_midpoint(
                coords['ls'], coords['rs'], self.config.min_keypoint_confidence
            )
            mid_hip = self._calculate_midpoint(
                coords['lh'], coords['rh'], self.config.min_keypoint_confidence
            )

            if mid_hip is None:
                return features

            # Calculate reference height for scaling
            reference_height = None
            if mid_shoulder is not None:
                reference_height = abs(mid_shoulder['y'] - mid_hip['y'])
                if reference_height < 1e-5:
                    reference_height = None

            # Normalize all keypoints
            for name in self.keypoint_data.names:
                x_idx, y_idx, _ = self.get_keypoint_indices(name)

                # Translate relative to hip center
                normalized[x_idx] -= mid_hip['x']
                normalized[y_idx] -= mid_hip['y']

                # Scale by reference height if available
                if reference_height is not None:
                    normalized[x_idx] /= reference_height
                    normalized[y_idx] /= reference_height

        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            return features

        return normalized

    def _calculate_midpoint(self, p1: Dict, p2: Dict, min_confidence: float) -> Optional[Dict]:
        """Calculate midpoint between two keypoints if confidence is sufficient."""
        valid_p1 = p1['c'] > min_confidence
        valid_p2 = p2['c'] > min_confidence

        if valid_p1 and valid_p2:
            return {'x': (p1['x'] + p2['x']) / 2, 'y': (p1['y'] + p2['y']) / 2}
        elif valid_p1:
            return {'x': p1['x'], 'y': p1['y']}
        elif valid_p2:
            return {'x': p2['x'], 'y': p2['y']}
        else:
            return None


class FallDetector:
    """Main fall detection system with enhanced alarm functionality."""

    def __init__(self, config: Config):
        self.config = config
        self.model_manager = ModelManager(config)
        self.feature_extractor = FeatureExtractor(config)
        self.audio_manager = AudioManager(config)
        self.feature_sequence = deque(maxlen=config.input_timesteps)
        self.last_fall_time = 0.0
        self.fall_events = []

        # Enhanced fall detection state
        self.consecutive_falls = 0
        self.current_fall_state = False
        self.sequence_ready = False

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = None

    def initialize(self) -> bool:
        """Initialize the detection system."""
        if not self.model_manager.load_model():
            return False

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.config.pose_complexity,
            smooth_landmarks=True,
            min_detection_confidence=self.config.detection_confidence,
            min_tracking_confidence=self.config.tracking_confidence
        )

        return True

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[str, float, bool, Dict]:
        """
        Process a single frame for fall detection.

        Returns:
            Tuple of (status, confidence, pose_detected, detection_info)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Get pose landmarks
        results = self.pose.process(rgb_frame)
        pose_detected = results.pose_landmarks is not None

        # Extract features
        features = self.feature_extractor.extract_features(results)
        self.feature_sequence.append(features)

        # Check if sequence is ready
        self.sequence_ready = len(self.feature_sequence) == self.config.input_timesteps

        detection_info = {
            'consecutive_falls': self.consecutive_falls,
            'current_fall_state': self.current_fall_state,
            'alarm_active': self.audio_manager.alarm_active,
            'sequence_ready': self.sequence_ready,
            'frames_needed': self.config.input_timesteps - len(self.feature_sequence)
        }

        # Make prediction if sequence is full
        if self.sequence_ready:
            sequence_array = np.array(self.feature_sequence, dtype=np.float32)
            fall_probability = self.model_manager.predict(sequence_array)

            if fall_probability >= self.config.fall_threshold:
                self.consecutive_falls += 1
                self.current_fall_state = True

                # Check if we have enough consecutive falls
                if self.consecutive_falls >= self.config.min_consecutive_falls:
                    # Check cooldown period
                    if timestamp - self.last_fall_time > self.config.fall_cooldown:
                        event_msg = f"FALL DETECTED at {timestamp:.1f}s (Confidence: {fall_probability:.3f})"
                        logger.warning(event_msg)
                        self.fall_events.append(event_msg)
                        self.last_fall_time = timestamp

                        # Start alarm
                        self.audio_manager.start_alarm()

                    return "FALL_DETECTED", fall_probability, pose_detected, detection_info
                else:
                    return "FALL_DETECTED", fall_probability, pose_detected, detection_info
            else:
                # Reset consecutive fall counter if no fall detected
                self.consecutive_falls = 0
                self.current_fall_state = False
                return "NORMAL", 1.0 - fall_probability, pose_detected, detection_info
        else:
            return "COLLECTING", 0.0, pose_detected, detection_info

    def stop_alarm_manual(self):
        """Manually stop the alarm (called when user presses space)."""
        self.audio_manager.stop_alarm()
        logger.info("Alarm manually stopped by user")

    def get_results_summary(self) -> Dict:
        """Get detection results summary."""
        return {
            'total_falls': len(self.fall_events),
            'fall_events': self.fall_events.copy()
        }

    def cleanup(self):
        """Clean up resources."""
        self.audio_manager.stop_alarm()
        if self.pose:
            self.pose.close()


class VideoProcessor:
    """Handles video processing and visualization."""

    def __init__(self, config: Config):
        self.config = config
        self.detector = FallDetector(config)

    def process_video(self, video_path: str, show_realtime: bool = True, save_output: bool = False) -> bool:
        """Process video file for fall detection."""
        if not self.detector.initialize():
            logger.error("Failed to initialize fall detector")
            return False

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {width}x{height} at {fps:.1f} FPS, {total_frames} frames")

        # Setup video writer
        writer = None
        if save_output:
            output_path = Path(video_path).with_name(f"processed_{Path(video_path).name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            logger.info(f"Output will be saved to: {output_path}")

        # Process frames
        frame_count = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                timestamp = frame_count / fps

                # Process frame
                status, confidence, pose_detected, detection_info = self.detector.process_frame(frame, timestamp)

                # Create visualization
                display_frame = self._create_enhanced_visualization(
                    frame, status, confidence, pose_detected, frame_count, timestamp, detection_info
                )

                # Save frame if requested
                if writer:
                    writer.write(display_frame)

                # Show real-time preview
                if show_realtime:
                    cv2.imshow('Fall Detection - Press SPACE to stop alarm, Q to quit', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested quit")
                        break
                    elif key == ord(' '):  # Space key to stop alarm
                        self.detector.stop_alarm_manual()

                # Progress logging
                if frame_count % 100 == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            return False

        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if show_realtime:
                cv2.destroyAllWindows()
            self.detector.cleanup()

            # Print summary
            processing_time = time.time() - start_time
            results = self.detector.get_results_summary()

            logger.info(f"Processing complete in {processing_time:.2f}s")
            logger.info(f"Processed {frame_count} frames")
            logger.info(f"Detected {results['total_falls']} fall events")

            for event in results['fall_events']:
                logger.info(f"  - {event}")

        return True

    def _create_enhanced_visualization(self, frame: np.ndarray, status: str, confidence: float,
                                       pose_detected: bool, frame_count: int, timestamp: float,
                                       detection_info: Dict) -> np.ndarray:
        """Create enhanced visualization with alarm and accuracy information."""
        vis_frame = frame.copy()

        # Colors for different states
        colors = {
            'FALL_DETECTED': (0, 0, 255),  # Red
            'NORMAL': (0, 255, 0),  # Green
            'COLLECTING': (0, 255, 255)  # Yellow
        }

        font = cv2.FONT_HERSHEY_DUPLEX
        small_font = cv2.FONT_HERSHEY_SIMPLEX

        # Main status (top-right)
        status_text = f"{status} ({confidence:.3f})" if confidence > 0 else status
        text_color = colors.get(status, (255, 255, 255))

        text_size = cv2.getTextSize(status_text, font, 0.8, 2)[0]
        text_pos = (frame.shape[1] - text_size[0] - 20, 40)
        cv2.putText(vis_frame, status_text, text_pos, font, 0.8, text_color, 2, cv2.LINE_AA)

        # Consecutive falls counter (below status)
        if detection_info['consecutive_falls'] > 0:
            consecutive_text = f"Consecutive: {detection_info['consecutive_falls']}/{self.config.min_consecutive_falls}"
            consecutive_color = (0, 165, 255) if detection_info[
                                                     'consecutive_falls'] < self.config.min_consecutive_falls else (
            0, 0, 255)
            cv2.putText(vis_frame, consecutive_text,
                        (frame.shape[1] - 300, 70), small_font, 0.6, consecutive_color, 2)

        # Alarm status and alert
        if detection_info.get('alarm_active', False):
            # Flashing alarm text
            flash_intensity = int(abs(np.sin(time.time() * 8)) * 255)  # Fast flashing
            alarm_color = (flash_intensity, flash_intensity, 255)

            # Large alarm text
            alarm_text = "FALL ALARM!"
            alarm_size = cv2.getTextSize(alarm_text, font, 1.2, 3)[0]
            alarm_pos = (frame.shape[1] // 2 - alarm_size[0] // 2, 100)
            cv2.putText(vis_frame, alarm_text, alarm_pos, font, 1.2, alarm_color, 3, cv2.LINE_AA)

            # Instructions
            instruction_text = "Press SPACE to stop alarm"
            inst_size = cv2.getTextSize(instruction_text, small_font, 0.6, 1)[0]
            inst_pos = (frame.shape[1] // 2 - inst_size[0] // 2, 130)
            cv2.putText(vis_frame, instruction_text, inst_pos, small_font, 0.6, (255, 255, 255), 1)

        # Current fall state indicator
        if detection_info.get('current_fall_state', False):
            state_text = "FALL STATE: ACTIVE"
            cv2.putText(vis_frame, state_text, (20, 100), small_font, 0.7, (0, 0, 255), 2)

        # Frame info (top-left)
        info_text = f"Frame: {frame_count} | Time: {timestamp:.1f}s"
        cv2.putText(vis_frame, info_text, (20, 30), small_font, 0.6, (255, 255, 255), 1)

        # Sequence progress (top-left, second line)
        if not detection_info['sequence_ready']:
            frames_needed = detection_info.get('frames_needed', 0)
            progress_text = f"Initializing: {self.config.input_timesteps - frames_needed}/{self.config.input_timesteps}"
            cv2.putText(vis_frame, progress_text, (20, 55), small_font, 0.5, (255, 255, 0), 1)

        # Pose detection status (bottom-left)
        pose_text = "Pose: Detected" if pose_detected else "Pose: Not Detected"
        pose_color = (0, 255, 0) if pose_detected else (0, 0, 255)
        cv2.putText(vis_frame, pose_text, (20, frame.shape[0] - 45), small_font, 0.6, pose_color, 1)

        # Accuracy info (bottom-left, second line)
        accuracy_text = f"Accuracy: {self.config.min_consecutive_falls} frames required"
        cv2.putText(vis_frame, accuracy_text, (20, frame.shape[0] - 20), small_font, 0.5, (200, 200, 200), 1)

        # Detection threshold info (bottom-right)
        threshold_text = f"Threshold: {self.config.fall_threshold:.2f}"
        threshold_size = cv2.getTextSize(threshold_text, small_font, 0.5, 1)[0]
        threshold_pos = (frame.shape[1] - threshold_size[0] - 20, frame.shape[0] - 20)
        cv2.putText(vis_frame, threshold_text, threshold_pos, small_font, 0.5, (200, 200, 200), 1)

        return vis_frame


def main():
    """Main function with enhanced configuration options."""
    print("=" * 60)
    print("🚨 ADVANCED FALL DETECTION SYSTEM WITH ALARM 🚨")
    print("=" * 60)
    print("Features:")
    print("- Audio alarm when falls are detected")
    print("- Requires multiple consecutive frames for better accuracy")
    print("- Real-time visual and audio alerts")
    print("- Press SPACE to manually stop alarm")
    print("- Press Q to quit")
    print("=" * 60)

    # Get user input
    video_path = input("Enter video file path: ").strip()
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return

    show_realtime = input("Show real-time preview? (y/n, default: y): ").strip().lower() != 'n'
    save_output = input("Save processed video? (y/n, default: n): ").strip().lower() == 'y'

    # Advanced configuration options
    print("\n🔧 Advanced Settings (press Enter for defaults):")

    config = Config()

    # Allow user to customize key parameters
    min_frames_input = input(
        f"Minimum consecutive frames for fall confirmation (default: {config.min_consecutive_falls}): ").strip()
    if min_frames_input.isdigit():
        config.min_consecutive_falls = max(1, min(20, int(min_frames_input)))
        print(f"✓ Set to {config.min_consecutive_falls} frames")

    threshold_input = input(f"Fall detection threshold (0.0-1.0, default: {config.fall_threshold}): ").strip()
    try:
        threshold = float(threshold_input)
        if 0.0 <= threshold <= 1.0:
            config.fall_threshold = threshold
            print(f"✓ Set to {config.fall_threshold}")
    except ValueError:
        pass

    alarm_duration_input = input(f"Alarm duration in seconds (default: {config.alarm_duration}): ").strip()
    try:
        duration = float(alarm_duration_input)
        if 1.0 <= duration <= 30.0:
            config.alarm_duration = duration
            print(f"✓ Set to {config.alarm_duration} seconds")
    except ValueError:
        pass

    cooldown_input = input(f"Fall event cooldown in seconds (default: {config.fall_cooldown}): ").strip()
    try:
        cooldown = float(cooldown_input)
        if 1.0 <= cooldown <= 60.0:
            config.fall_cooldown = cooldown
            print(f"✓ Set to {config.fall_cooldown} seconds")
    except ValueError:
        pass

    print("\n📋 Configuration Summary:")
    print(f"  • Video: {Path(video_path).name}")
    print(f"  • Consecutive frames required: {config.min_consecutive_falls}")
    print(f"  • Detection threshold: {config.fall_threshold}")
    print(f"  • Alarm duration: {config.alarm_duration}s")
    print(f"  • Event cooldown: {config.fall_cooldown}s")
    print(f"  • Real-time preview: {show_realtime}")
    print(f"  • Save output: {save_output}")
    print(f"  • Audio available: {AUDIO_AVAILABLE}")

    if not Path(config.model_path).exists():
        print(f"\n⚠️  Warning: Model file not found: {config.model_path}")
        print("Please ensure the TFLite model file is in the correct location.")
        return

    print("\n🚀 Starting fall detection system...")
    print("-" * 60)

    # Process video
    processor = VideoProcessor(config)

    try:
        success = processor.process_video(video_path, show_realtime, save_output)

        if success:
            print("\n✅ Processing completed successfully!")
            print("\n💡 Tips:")
            print("  - Adjust threshold if getting too many/few detections")
            print("  - Increase consecutive frames for higher accuracy")
            print("  - Use good lighting and clear pose visibility for best results")
            if AUDIO_AVAILABLE:
                print("  - Audio alarms are enabled - press SPACE to stop them")
            else:
                print("  - Audio alarms disabled - install pygame for audio support")
        else:
            print("\n❌ Processing failed!")
            print("Check the logs above for error details.")

    except KeyboardInterrupt:
        print("\n\n⏹️  Processing interrupted by user")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error in main: {e}")

    finally:
        print("\n🔚 Fall detection system shutdown complete.")


if __name__ == "__main__":
    main()