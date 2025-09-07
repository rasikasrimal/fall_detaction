# Test Cases - Fall Detection

**Status:** Assumption ⚠️

| ID | Title | Preconditions | Steps | Expected |
|----|-------|--------------|-------|---------|
| TC1 | Config defaults | None | Load `Config` dataclass | Fields match defaults |
| TC2 | Feature sequence length | Model loaded | Append features until full | Prediction executed |
