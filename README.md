# Fall Detection

This repository provides two Python scripts that run fall-detection models on pose keypoints captured with [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose).  The models raise an audio/visual alarm when a fall is detected in a video stream.

## Features

- Pose extraction with OpenCV and MediaPipe.
- Two inference backends:
  - **TensorFlow Lite Transformer** (`run_tf.py`).
  - **PyTorch LSTM or TensorFlow Lite** (`tf_lstm.py`).  The script automatically offers the available model files and frameworks.
- Optional audio alarm using `pygame`.
- Configurable number of consecutive frames, detection threshold, and alarm duration.

Model files `fall_transformer.tflite` and `fall_lstm_model.pth` are included for convenience.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The project requires either `tflite-runtime` or full `tensorflow` for TensorFlow Lite support and `torch` for the PyTorch LSTM model.  `pygame` enables the optional alarm sound.

## Usage

### TensorFlow Lite transformer

```bash
python run_tf.py
```

The script prompts for the path to a video file and optional runtime settings.  It displays the processed video and triggers an alarm when a fall is detected.

### Unified interface (TensorFlow Lite or PyTorch)

```bash
python tf_lstm.py
```

This script lets you choose between the included TensorFlow Lite transformer model or a PyTorch LSTM model.  Configuration questions guide you through tuning detection parameters.

Press **Q** to quit the video window.  Press **SPACE** to stop the alarm.

## License

This repository does not include an explicit license.  Use at your own risk.

