# Project Diagrams

## Index
- [Data Flow Diagram](#data-flow-diagram)
- [BPMN 2.0 Diagram](#bpmn-20-diagram)
- [Architectural Diagram](#architectural-diagram)
- [Sequence Diagram](#sequence-diagram)
- [Use Case Diagram](#use-case-diagram)
- [Entity Relationship Diagram](#entity-relationship-diagram)
- [Additional Diagram Suggestions](#additional-diagram-suggestions)

## Data Flow Diagram
```mermaid
flowchart TD
    User([User]) --> CLI[CLI Interface]
    CLI --> Video[Video Frames]
    Video --> Pose[MediaPipe Pose]
    Pose --> Normalize[Feature Normalization]
    Normalize --> Model[ML Model (TFLite/PyTorch)]
    Model --> Decision{Fall Detected?}
    Decision -->|Yes| Alarm[Visual/Audio Alarm]
    Decision -->|No| Loop[Continue Processing]
    Alarm --> Loop
```

## BPMN 2.0 Diagram
```mermaid
flowchart TD
    Start((Start)) --> Capture[Capture Frame]
    Capture --> Extract[Extract Pose]
    Extract --> Append[Append Features]
    Append --> SeqReady{Sequence Ready?}
    SeqReady -->|No| Capture
    SeqReady -->|Yes| Evaluate[Run Model]
    Evaluate --> Fall{Fall?}
    Fall -->|No| Capture
    Fall -->|Yes| Alarm[Trigger Alarm]
    Alarm --> End((End))
```

## Architectural Diagram
```mermaid
flowchart LR
    subgraph UserLayer
        U[User]
    end
    subgraph Processing
        CLI[CLI]
        Pose[MediaPipe Pose]
        Model[ML Models]
        Alarm[Alarm Handler]
    end
    U --> CLI
    CLI --> Pose
    Pose --> CLI
    CLI --> Model
    Model --> CLI
    CLI --> Alarm
```

## Sequence Diagram
```mermaid
sequenceDiagram
    participant U as User
    participant C as CLI
    participant P as MediaPipe
    participant M as Model
    participant A as Alarm
    U->>C: Start program
    C->>U: Request video path
    U-->>C: Provide path
    C->>P: Process frame
    P-->>C: Pose landmarks
    C->>M: Evaluate features
    M-->>C: Fall probability
    alt Fall above threshold
        C->>A: Trigger alarm
        A-->>U: Alert
    end
```

## Use Case Diagram
```mermaid
flowchart TD
    user((User)) --> fallDetection[Fall Detection CLI]
    fallDetection --> alarm[(Alarm Output)]
```

## Entity Relationship Diagram
The system does not persist data, so no entity relationship diagram is required.

## Additional Diagram Suggestions
- Deployment diagram showing packaging for different environments
- State diagram for alarm activation and cooldown
- Component diagram outlining module boundaries
