# Executive Brief: LLM Fraud Classification Pilot Study

## Overview & Scope
We ran an n=50 pilot (10 transactions per category, the same 50 for every configuration) to evaluate Large Language Models (LLMs) for real-time transaction fraud classification. The goal was to understand if LLMs could accurately classify transactions into 5 categories from narratives built on categorical and behavioral signals, and whether their latency supports synchronous blocking.

## Key Findings
1. **LLMs are too slow for synchronous blocking.** The fastest configuration (GPT-4o-mini, Few-Shot) averaged 531ms per transaction. Our synchronous blocking layer requires a sub-200ms response time.
2. **Zero-Shot vs. Few-Shot:** LLMs need explicit examples to apply specialized fraud definitions. Adding 2 examples (Few-Shot) raised accuracy from 74% to 90% on GPT-4o-mini and from 74% to 92% on Claude Haiku 4.5. Peak accuracy: 92% (Claude Haiku 4.5).
3. **Chain-of-Thought (CoT) degrades performance:** Asking the model to reason step by step before classifying raised average latency to as much as 6458ms, multiplied cost, and *reduced* accuracy compared with Few-Shot (56% on GPT-4o-mini, 74% on Claude Haiku 4.5). The scratchpads show the models overthinking secondary signals. Caveat: the CoT prompt listed only the category names, without the definitions or examples, so part of the drop comes from that and not from step-by-step reasoning alone.

## Recommendations & Next Steps
1. **Do not deploy LLMs in the synchronous path.**
2. **Proceed to Traditional ML:** Use the insights from this pilot to engineer features for a traditional Machine Learning model (e.g., XGBoost or Random Forest), which can meet the sub-200ms latency requirement.
3. **Expand Evaluation:** Scale testing from 50 samples to a larger dataset (10,000+) to train and validate the traditional ML layer.
4. **Potential Asynchronous Use Case:** The Few-Shot prompt can run as an asynchronous "shadow" pipeline that writes human-readable narratives for the fraud review team after the ML model blocks a transaction.
