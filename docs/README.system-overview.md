# System Overview

**Status:** Derived from Code ✅

## Executive Summary
A command-line fall detection utility processes video frames, extracts human pose keypoints using MediaPipe, and evaluates sequences with either a TensorFlow Lite transformer or PyTorch LSTM model. When a fall is detected for a configurable number of consecutive frames, the system raises a visual and optional audio alarm.

## System Context (C4-1)
- **User** interacts via CLI to supply video input.
- **MediaPipe Pose** library provides pose landmarks.
- **ML Models** (TFLite or PyTorch) score pose sequences for fall probability.
- **Audio Output** uses `pygame` when available to play alarms.

## Key Capabilities
- Pose keypoint mapping for 17 landmarks.
- Feature normalization relative to hip and shoulder midpoints.
- Sequence-based fall classification with alarm cooldown and duration controls.

## Domain Glossary
| Term | Definition |
|------|------------|
| *Pose Landmark* | 3D point (x,y,visibility) returned by MediaPipe. |
| *Fall Event* | Condition where model confidence exceeds threshold for required consecutive frames. |
| *Cooldown* | Minimum time between recorded fall events. |

## Tech Stack
| Layer | Technology |
|-------|------------|
| Language | Python |
| ML Frameworks | TensorFlow Lite, PyTorch |
| CV Library | OpenCV |
| Pose Estimation | MediaPipe Pose |
| Audio | pygame |
| CI/CD | none (manual) |
