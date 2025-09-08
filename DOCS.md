# Combined Documentation


## Documentation Index

- [Changelog](#changelog)
- [System Overview](#system-overview)
- [High-Level Diagrams](#high-level-diagrams)
- [Error Catalog](#error-catalog)
- [RPC / GraphQL](#rpc--graphql)
- [Webhooks & Integrations](#webhooks--integrations)
- [Assumptions & Open Questions](#assumptions--open-questions)
- [Glossary](#glossary)
- [Mapping Code to Docs](#mapping-code-to-docs)
- [Runtime Views](#runtime-views)
- [Schema Catalog](#schema-catalog)
- [Observability](#observability)
- [Release & CI/CD](#release--cicd)
- [Runbook](#runbook)
- [Compliance Notes](#compliance-notes)
- [Secrets & Identity](#secrets--identity)
- [Threat Model](#threat-model)
- [E2E Tests](#e2e-tests)
- [Load & Resilience](#load--resilience)
- [Test Cases - Fall Detection](#test-cases---fall-detection)
- [Test Strategy](#test-strategy)

## Changelog

# Changelog

- 2024-05-13: Initial documentation and test scaffold generated.



## File: README.system-overview.md

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

# High-Level Diagrams

- [Data Flow Diagram (Level 2)](diagrams/dfd-level2-fall-detection.mmd)
- [BPMN 2.0 Process Diagram](diagrams/fall-detection.bpmn)
- [Entity-Relationship Diagram](diagrams/er.mmd)

### Suggested Additional Diagrams
- C4 Context and Container diagrams outlining system boundaries and components.
- Sequence diagrams for key fall detection workflows.
- State diagrams representing fall state transitions.
- Deployment diagrams for local versus distributed environments.

## File: apis/error-catalog.md

# Error Catalog

**Status:** Derived from Code ✅

| Code | Message/Condition | Reference |
|------|------------------|-----------|
| `FALL_DETECTED` | Model probability >= threshold for required frames | [run_tf.py](run_tf.py) |
| `NORMAL` | Model probability below threshold | [run_tf.py](run_tf.py) |
| `COLLECTING` | Insufficient frames to evaluate | [run_tf.py](run_tf.py) |



## File: apis/rpc-graph-ql.md

# RPC / GraphQL

**Status:** Assumption ⚠️

The repository exposes no RPC, gRPC, or GraphQL interfaces.



## File: apis/webhooks-integrations.md

# Webhooks & Integrations

**Status:** Assumption ⚠️

No external webhooks or third-party integrations are present.



## File: appendices/assumptions-and-open-questions.md

# Assumptions & Open Questions

| Item | Verification |
|------|-------------|
| Model accuracy thresholds tuned for provided weights | Collect evaluation dataset |
| Alarm sound generation adequate for accessibility | User testing |
| No persistence beyond runtime | Confirm no logs stored externally |



## File: appendices/glossary.md

# Glossary

| Term | Meaning |
|------|--------|
| TFLite | TensorFlow Lite runtime |
| LSTM | Long Short-Term Memory neural network |
| Pose Landmark | Body keypoint from MediaPipe |



## File: appendices/mapping-code-to-docs.md

# Mapping Code to Docs

| Code File | Documentation |
|-----------|---------------|
| `run_tf.py` | Architecture diagrams, Error catalog, Runbook |
| `tf_lstm.py` | Architecture diagrams, Runbook |



## File: architecture/runtime-views.md

# Runtime Views

**Status:** Derived from Code ✅

- **VideoProcessor** orchestrates frame capture and delegates per-frame analysis to `FallDetector`.
- **FallDetector** maintains a deque of normalized pose features, calls `ModelManager.predict` when enough frames are collected, and triggers `AudioManager.start_alarm` on high confidence.
- **ModelManager** loads a TFLite model and enforces input shape validation before inference.
- **AudioManager** plays a generated sine-wave alarm in a thread until duration elapses or `stop_alarm()` is invoked.
- Logging uses Python's standard library; no caching or external concurrency primitives beyond threading.



## File: data/schema-catalog.md

# Schema Catalog

**Status:** Derived from Code ✅

| Name | Fields | Notes |
|------|--------|-------|
| `KeypointData` | 17 landmarks × (x,y,visibility) | Indices defined in `run_tf.py` and `tf_lstm.py` | 
| `Config` | model_path, input_timesteps, fall_threshold, etc. | Runtime tuning parameters |
| `FeatureSequence` | deque of normalized keypoints | `maxlen` equals `input_timesteps` |



## File: operations/observability.md

# Observability

**Status:** Derived from Code ✅

- Logging via `logging` module at INFO level by default.
- Progress messages every 100 frames.
- Fall events appended to an in-memory list and logged with `WARNING` level.



## File: operations/release-and-cicd.md

# Release & CI/CD

**Status:** Assumption ⚠️

No automated pipeline; releases consist of committing scripts and model files.
Recommended steps: lint, run unit tests, validate on sample video before tagging.



## File: operations/runbook.md

# Runbook

**Status:** Derived from Code ✅

1. Install dependencies: `pip install -r requirements.txt`.
2. Run `python run_tf.py` or `python tf_lstm.py`.
3. Provide path to video when prompted.
4. Press SPACE to stop alarm, Q to quit.





## File: security/compliance-notes.md

# Compliance Notes

**Status:** Assumption ⚠️

- Video frames may contain PII; ensure local processing and retention policies.
- No explicit data retention; user responsible for deleting outputs.



## File: security/secrets-and-identity.md

# Secrets & Identity

**Status:** Assumption ⚠️

- No authentication or authorization mechanisms.
- Model files stored locally; protect filesystem permissions.
- Audio playback requires no credentials.



## File: security/threat-model.md

# Threat Model

**Status:** Assumption ⚠️

| STRIDE | Considerations | Mitigations |
|--------|----------------|-------------|
| Spoofing | User supplies malformed video | rely on MediaPipe validation |
| Tampering | Model files could be replaced | verify file paths and hashes |
| Repudiation | CLI logs events with timestamps | keep log files |
| Information Disclosure | No network services; data stays local | ensure local storage security |
| Denial of Service | Large video may exhaust memory | limit frame size/length |
| Elevation of Privilege | Not applicable to local CLI | run under least privilege |



## File: testing/e2e/README.md

# E2E Tests

Placeholder for future end-to-end scripts using short test videos.



## File: testing/load-and-resilience.md

# Load & Resilience

**Status:** Assumption ⚠️

- Target 25 FPS processing on 720p video.
- Alarm thread should not block main loop.
- No chaos tests defined.



## File: testing/test-cases/fall-detection.md

# Test Cases - Fall Detection

**Status:** Assumption ⚠️

| ID | Title | Preconditions | Steps | Expected |
|----|-------|--------------|-------|---------|
| TC1 | Config defaults | None | Load `Config` dataclass | Fields match defaults |
| TC2 | Feature sequence length | Model loaded | Append features until full | Prediction executed |



## File: testing/test-strategy.md

# Test Strategy

**Status:** Assumption ⚠️

- **Unit**: parse configuration, verify keypoint mappings.
- **Integration**: mock feature sequences and ensure `ModelManager` shape checks.
- **E2E**: run CLI on short video and assert alarm behavior (manual).
- Target coverage: 60% lines for utility functions.

