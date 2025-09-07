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
