# Chain-of-thought errors, one by one
All 35 wrong answers from the two chain-of-thought runs on the 50-transaction sample (the prompt without category definitions), with the reason for each, read from the models' scratchpads in `llm_eval_log_cot_openai.csv` and `llm_eval_log_cot_anthropic.csv`. The few-shot + chain-of-thought runs added in v1.1.0 are not broken down here.

## Claude Haiku 4.5 (13 errors)

### Carding predicted as BIN_Testing (8)
* **Transactions:** `tx_9bf5f27eda`, `tx_dde01d0e34`, `tx_ee4392984f`, `tx_5509baab1c`, `tx_75fb59fa34`, `tx_4923b23013`, `tx_49cb510cc6`, `tx_6bda68cdfd`
* **Root cause:** The scratchpads treat "rotated through N different cards" as the defining sign of BIN testing. They mention the CVV match and issuer decline (Code 51) but read it as part of the testing, not as proof the attacker already had the card details. Several scratchpads even note that Carding could apply, then pick BIN_Testing as "more specific". The CoT prompt lists only the category names, not their definitions, so nothing ties CVV match to Carding.

### Stolen_Card predicted as Legitimate (2)
* **Transactions:** `tx_670e2f128b`, `tx_baa510e4cf`
* **Root cause:** Noisy stolen-card records: billing and shipping ZIPs match, no proxy, normal velocity, established account. The narrative carries no fraud signal, and the scratchpads describe it correctly. No prompt can recover these; the signal is lost when the raw fields become narrative.

### Legitimate predicted as Stolen_Card (1)
* **Transactions:** `tx_8a2c238094`
* **Root cause:** A noisy legitimate order (brand-new guest checkout, proxy IP, shipping ZIP different from billing). The scratchpad reads the passed CVV/AVS checks as proof the fraudster holds full card details. From the narrative alone this is hard to tell apart from a stolen card; a gift shipped to another ZIP looks the same.

### Stolen_Card predicted as Carding (1)
* **Transactions:** `tx_73212d1b7b`
* **Root cause:** Established account, clean authorization, drop-house shipping. The scratchpad rules out Stolen_Card because verification passed, then calls it Carding, meaning a stolen card buying goods for a drop-house. Without definitions in the prompt, the model's own meaning of Carding wins.

### Synthetic_Identity predicted as Legitimate (1)
* **Transactions:** `tx_b1f4d05a77`
* **Root cause:** A "sleeper" synthetic account (245 days old, gmail.com, synthetic score 0.508). For accounts 14 days or older the identity narrative only says "Established user account" and drops the score, so the model never sees it.

## GPT-4o-mini (22 errors)

### BIN_Testing predicted as Carding (8)
* **Transactions:** `tx_144223243b`, `tx_1c846e38a9`, `tx_8244e90dcf`, `tx_6d88b819c8`, `tx_ea22d9dac6`, `tx_c4ae0baac3`, `tx_fd40f6267b`, `tx_a4c84423a9`
* **Root cause:** The scratchpads list every BIN-testing signal correctly (N7 CVV mismatch, 20-58 card numbers in 60 seconds on one BIN, pre-auth or micro amount) and then call it Carding. The model uses "carding" as the umbrella term for any card-testing attack; the CoT prompt gives no definitions to separate the two.

### Carding predicted as Synthetic_Identity (8)
* **Transactions:** `tx_3bc8b693f1`, `tx_dde01d0e34`, `tx_fcfd5e7d09`, `tx_5509baab1c`, `tx_75fb59fa34`, `tx_4923b23013`, `tx_49cb510cc6`, `tx_6bda68cdfd`
* **Root cause:** Identity fixation: the scratchpads focus on the 1-2 day old account and the disposable email domain and conclude Synthetic_Identity, giving little weight to the 9-20 card rotation and the CVV-matched issuer decline that define Carding.

### Stolen_Card predicted as Carding (2)
* **Transactions:** `tx_47a3122388`, `tx_5008de09de`
* **Root cause:** Established accounts with clean authorizations and drop-house shipping. The scratchpads spot the drop-house pattern and label it Carding, again using the term for any purchase with stolen card details.

### Stolen_Card predicted as Legitimate (2)
* **Transactions:** `tx_670e2f128b`, `tx_baa510e4cf`
* **Root cause:** Same records as the Claude Haiku 4.5 errors: the narrative has no fraud signal, and the scratchpads read it correctly.

### Legitimate predicted as Stolen_Card (1)
* **Transactions:** `tx_8a2c238094`
* **Root cause:** Same noisy legitimate order as in the Claude Haiku 4.5 section. The scratchpad sees clean authorization and normal velocity but lets the billing/shipping mismatch decide.

### Synthetic_Identity predicted as Legitimate (1)
* **Transactions:** `tx_b1f4d05a77`
* **Root cause:** Same sleeper account as the Claude Haiku 4.5 error: the synthetic identity score is not in the narrative.

## What this means for the next model
When two signals pointed different ways (high card velocity but a brand-new account, say), the models did not weigh them the same way twice. The raw signals (account age, velocity count, CVV match, ZIP mismatch) belong in a table as features, so a model such as XGBoost can learn how they combine.
