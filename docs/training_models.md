## Training models

| Model                           | Input | Train?   | Why                   |
| ------------------------------- | ----- | -------- | --------------------- |
| Depression classifier           | Text  | ✅       | Custom mental health  |
| Anxiety classifier              | Text  | ✅       | Custom mental health  |
| Stress classifier               | Text  | Optional | Derived but can train |
| Suicide risk classifier         | Text  | ✅ MUST  | No pretrained model   |
| Cognitive distortion classifier | Text  | ✅ MUST  | No pretrained         |
| Conversation tone classifier    | Text  | Optional | Your choice           |

You will train 4 models only:

- Depression-from-text
- Anxiety-from-text
- Suicide risk
- Cognitive distortion

Optional: stress, conversation tone

You will NOT train:

- Face emotion
- Speech emotion
- Speech features
- Fusion model
- Recommendation engine
- Alerts