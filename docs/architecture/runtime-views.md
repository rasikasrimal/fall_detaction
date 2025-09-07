# Runtime Views

**Status:** Derived from Code ✅

- **VideoProcessor** orchestrates frame capture and delegates per-frame analysis to `FallDetector`.
- **FallDetector** maintains a deque of normalized pose features, calls `ModelManager.predict` when enough frames are collected, and triggers `AudioManager.start_alarm` on high confidence.
- **ModelManager** loads a TFLite model and enforces input shape validation before inference.
- **AudioManager** plays a generated sine-wave alarm in a thread until duration elapses or `stop_alarm()` is invoked.
- Logging uses Python's standard library; no caching or external concurrency primitives beyond threading.
