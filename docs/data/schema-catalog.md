# Schema Catalog

**Status:** Derived from Code ✅

| Name | Fields | Notes |
|------|--------|-------|
| `KeypointData` | 17 landmarks × (x,y,visibility) | Indices defined in `run_tf.py` and `tf_lstm.py` | 
| `Config` | model_path, input_timesteps, fall_threshold, etc. | Runtime tuning parameters |
| `FeatureSequence` | deque of normalized keypoints | `maxlen` equals `input_timesteps` |
