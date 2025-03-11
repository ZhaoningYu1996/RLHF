# RLHF

| **Method** | **Description**                                                     | **Status**         |
|------------|---------------------------------------------------------------------|--------------------|
| **PPO**    | Proximal Policy Optimization for fine-tuning language models       | Implemented        |
| **DPO**    | Direct Preference Optimization using reward signals from feedback  | Implemented        |
| **GRPO**   | Generalized Reinforcement Policy Optimization for LLM tuning       | Implemented        |
| *More*     | Other RLHF approaches planned as learning progresses               | *TBD*              |

## Overview

**RLHF** (Reinforcement Learning from Human Feedback) is a technique for aligning Large Language Models (LLMs) with desired behaviors by incorporating human feedback into the training process. This repository is **purely for learning and experimentation**, aiming to understand how various RLHF methods can be integrated with models like **LLaMA 3.2**.

## Getting Started

## Project Structure

```
.
├── data/              # Example or placeholder data
├── models/            # Model checkpoints or custom modules
├── train_ppo.py       # PPO training script
├── train_dpo.py       # DPO training script
├── train_grpo.py      # GRPO training script
├── requirements.txt
└── README.md
```

## License

This project is distributed under the [MIT License](LICENSE).

---

Feel free to open an issue or pull request if you have suggestions or questions!