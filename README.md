# LLM fraud classification: zero-shot vs. few-shot vs. chain-of-thought

We gave two AI models, GPT-4o-mini and Claude Haiku 4.5, the job of spotting card fraud. With good instructions they got 9 out of 10 payments right, about as good as a short list of hand-written rules on the same data. But they took between half a second and several seconds per payment, and a card payment can only wait about a fifth of a second. Then we changed how the fraud looked. The rules got none of the changed cases right, while the AI setups got between half and most of them.

Everything runs on made-up data, and every number on this page can be checked against the files in this repo. New to this? Read [Start here](#start-here-no-coding-needed) or [try it in your browser](#play-with-it-in-your-browser).

By Ricardo Iguaran, [Fortuenti Consulting](https://fortuenti.co). Version: see [`VERSION`](VERSION).

## Start here (no coding needed)

### What this is

When you pay with a card online, the store and the bank have a fraction of a second to decide whether the payment is fraud. This project asks whether an AI language model, the kind behind ChatGPT and Claude, can make that call: how often it is right, how long it takes and what it costs.

No real card, person or store is involved.

### How the experiment works

1. A script creates 5,000 fake card payments. 92% are normal purchases and the rest are four kinds of fraud. Each payment has about 30 technical fields: the amount, whether the card's security code matched, how many different cards the same connection tried in the last minute, and so on.
2. Language models read text, so each payment is rewritten as six short sentences, such as *"CVV matched, but the transaction was declined by the issuer."*
3. The same 50 payments go to two models, OpenAI's GPT-4o-mini and Anthropic's Claude Haiku 4.5, with different kinds of instructions (prompts). The model answers with one of five labels.
4. Each answer is compared with the true label, and the time and cost of every call are recorded.
5. A short list of if-then rules does the same job, as a baseline.

### The five labels

| Label | In plain words | A real example from the data |
|---|---|---|
| Legitimate | A normal purchase | A $137.18 order from a 689-day-old account. Card checks passed, shipped to the billing ZIP code. |
| BIN testing (card-number guessing) | A bot guessing card numbers until one works | A $0.50 "is this card valid?" check that failed the security code. The same connection tried 41 card numbers in one minute. |
| Carding (stolen-card checking) | A bot running through a list of stolen cards to see which still work | A $3.98 purchase where the security code was right but the bank declined it. 11 different cards tried in one minute from a brand-new session. |
| Stolen card | Someone uses one stolen card for a bigger purchase | An $887.02 order from a 273-day-old account, shipped to a different ZIP code than the card's billing address, through a proxy. |
| Synthetic identity | A fake person built from a mix of real and invented details | A $383.26 order from a 7-day-old account with a throwaway email address and a synthetic-identity score of 0.95 out of 1. |

These categories are simplified so each one has clear signals to learn from. Caveat 8 below explains how real fraud differs.

### Words you'll see

- **LLM (large language model):** the kind of AI that reads and writes text, like ChatGPT or Claude.
- **Prompt:** everything sent to the model: instructions, definitions and the payment description.
- **Zero-shot:** a prompt with the label definitions only.
- **Few-shot:** the same prompt plus a few worked examples. Here, two examples that explain how to tell BIN testing from carding.
- **Chain-of-thought (CoT):** asking the model to write out its reasoning before it answers. That written reasoning is the *scratchpad*, saved in the `cot_output` column of the CoT logs.
- **Temperature:** how much randomness the model uses. At 0 it almost always gives the same, most likely answer. Every run here uses 0.
- **Token:** a chunk of text, roughly three quarters of a word. Providers charge per token.
- **Latency:** how long one answer takes, in milliseconds (1,000 ms = 1 second). A decision to block a card payment usually has to come back in under 200 ms.
- **API and API key:** the way a program talks to OpenAI or Anthropic, and the password that charges your account for it. Never share a key. This project reads keys from a local `.env` file that is never uploaded.
- **Rule engine:** hand-written if-then rules, for example "more than 10 cards in a minute and the security code failed, so BIN testing".
- **Accuracy and recall:** accuracy is the share of all answers that are right. Recall for one label is the share of those cases the method caught.
- **Confusion matrix:** a table that counts what each true label was predicted as. It shows which labels get mixed up.
- **Card checks:** *CVV* is the 3-digit security code (4 digits on Amex). *BIN* is the first digits of the card number, which identify the bank that issued it: 6 in this data, 8 on newer cards. *AVS* checks the billing address. *Response codes* are the bank's answer: 00 approved, 05 do not honor, 51 insufficient funds, N7 security code mismatch.
- **Behavior and location:** *MCC* is the store's category code (5411 is a grocery store). *Velocity* is how many cards or attempts come from one connection in a short time. A *drop-house* is an address where criminals receive goods. A *proxy* hides where a connection really comes from.

### Look around without installing anything

GitHub shows CSV files as tables you can scroll and search. Good places to start:

- [`llm_eval_log_few-shot_anthropic.csv`](llm_eval_log_few-shot_anthropic.csv): each of the 50 payments, Claude's answer, whether it was right, how long it took and what it cost. The `prompt_sent` column is exactly what the model read.
- [`llm_eval_log_cot_openai.csv`](llm_eval_log_cot_openai.csv): GPT-4o-mini's written reasoning in `cot_output`. Type `False` in the search box to find the rows it got wrong and read how it talked itself into them.
- [`Engineer_Report.md`](Engineer_Report.md): every reasoning mistake, grouped by type, with the reason behind it.

### Play with it in your browser

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fortuenti-Consulting/llm-fraud-classification/blob/main/playground.ipynb)

[`playground.ipynb`](playground.ipynb) opens in Google Colab with a free Google account and nothing to install. Colab warns that the notebook was not written by Google; click *Run anyway*. Then click a gray code box and press Shift+Enter to run it, one box at a time. In a few minutes you can:

- read one payment as raw fields and as the text the model sees,
- run the rule engine and change one of its limits to see what it catches and what it misses,
- read a model's reasoning on a payment it got wrong,
- ask the model about a payment yourself, if you have an OpenAI developer account with a few dollars of prepaid credit (each question costs a fraction of a cent).

### Ideas to try next

These need Python on your computer and a code editor. Keep a copy of the project before you start, because some of them replace the published data.

- Add a third example to the few-shot prompt in [`classify_llm_few-shot.py`](classify_llm_few-shot.py) about gift orders shipped to another address. Today few-shot labels 0 of the 10 shifted gift orders as legitimate. Run `DATASET=shifted python3 classify_llm_few-shot.py` to see whether that changes (a few cents).
- Raise `NOISE_RATE` in [`dataset_generator.py`](dataset_generator.py) to make the data messier, then run `python3 dataset_generator.py && python3 card_fraud_analyzer.py` and see how the rules cope. This creates new transaction IDs, so the published model results no longer line up with the data.
- Change the wording of the descriptions in `enrich_for_llm` (in the same file), for example drop "consistent with drop-houses", to measure how much the wording steers the models. After regenerating the data, write 10 new transaction IDs per category into `eval_sample_ids.txt` and rerun the few-shot script.

## Results

In short: with two worked examples in the prompt (few-shot), both models get close to the best score possible on this data, but every setup takes roughly 3-32 times longer than the ~200 ms a card payment can wait. Asking the model to reason step by step (chain-of-thought) never improved accuracy on the original sample: without the category definitions it cost 18-34 points, and with them it landed within one transaction of few-shot while taking far more time and money. On a second set built to slip past the hand-written rules (which then score 0%), the four few-shot setups score 52-84%, and step-by-step reasoning helped Claude Haiku 4.5 (66% to 84%) but not GPT-4o-mini (62% to 52%).

Same 50 transactions for every run (10 per category, IDs in [`eval_sample_ids.txt`](eval_sample_ids.txt)), temperature 0 on both models. Each cell shows **accuracy · average time per answer · total cost of the 50 questions**, at list prices (GPT-4o-mini: $0.15 per million tokens read and $0.60 per million written; Claude Haiku 4.5: $1 and $5).

| Strategy | GPT-4o-mini | Claude Haiku 4.5 |
|---|---|---|
| Zero-shot | 74% · 595 ms · $0.0025 | 74% · 640 ms · $0.0206 |
| Few-shot (2 examples) | 90% · 531 ms · $0.0040 | **92%** · 664 ms · $0.0321 |
| Chain-of-thought | 56% · 2,045 ms · $0.0092 | 74% · 6,458 ms · $0.1649 |
| Rule engine (baseline) | 47/50 = 94% · <0.01 ms · $0 | |

Source: [`performance_grid_analysis.json`](performance_grid_analysis.json), [`confusion_matrix_analysis.json`](confusion_matrix_analysis.json), [`rule_engine_eval_log.csv`](rule_engine_eval_log.csv). On all 5,000 transactions the rule engine scores 99.5%.

What the errors show:

- **Zero-shot's main failure is Carding vs. BIN testing** (8-9 of 13 errors per model). Both are automated card attacks; the difference is in the verification signal (CVV mismatch = guessing numbers, CVV match plus issuer decline = stolen card lists). Two examples that spell out that difference remove all of them: few-shot has zero Carding/BIN testing errors on both models.
- **Three transactions are missed by every method, including the rule engine.** Two stolen-card records and one "sleeper" synthetic identity carry no fraud signal in their fields. That makes 47/50 (94%) the practical ceiling for this sample, and few-shot Claude Haiku 4.5 is one transaction below it.
- **Chain-of-thought hurt, but mostly because of the prompt.** The scratchpads show GPT-4o-mini using "Carding" as an umbrella term for any card attack and fixating on new accounts with disposable emails, and Claude Haiku 4.5 reading "rotated through many cards" as BIN testing. Adding the same checklist to the full few-shot prompt brings both models back to 90% (see [below](#shifted-stress-test-and-few-shot--cot)). Full breakdown with transaction IDs: [`Engineer_Report.md`](Engineer_Report.md). Summary for decision makers: [`VP_Report.md`](VP_Report.md).

## Caveats

Read these before quoting the numbers.

1. **Synthetic data.** All 5,000 transactions come from [`dataset_generator.py`](dataset_generator.py). No real cardholder, merchant or network data is involved. IPs, ZIP codes and user agents are random values from [Faker](https://faker.readthedocs.io/); BINs are random or well-known test-card prefixes.
2. **n = 50.** One transaction is 2 percentage points. Treat differences under ~10 points as noise.
3. **The rule engine is partly circular.** Its rules were written by the same person who wrote the generator, so it knows each category's signature. Its 94-99.5% shows that structured signals beat an LLM when they are this clean, not that rules would do this well on real traffic. The [shifted stress test](#shifted-stress-test-and-few-shot--cot) shows how quickly that breaks. Rule 4 in particular (different shipping ZIP on an older account means stolen card) would block every gift order in production; it is a teaching device.
4. **The chain-of-thought prompt is not a pure "add reasoning" test.** It lists the category names but not the definitions or examples the other two prompts have. The few-shot + CoT runs [below](#shifted-stress-test-and-few-shot--cot) isolate the effect of reasoning.
5. **Narratives lose information by design.** The LLMs read short text narratives built from the raw fields ([`enrich_for_llm`](dataset_generator.py)), not the fields themselves. For accounts 14 days or older, for example, the narrative drops the synthetic identity score.
6. **Latency was measured from a single machine** over the public APIs and includes network time.
7. **CoT output limits differ by model:** `max_tokens` is 300 for GPT-4o-mini and 800 for Claude Haiku 4.5. No response was cut off (every CoT output has its closing `</classification>` tag), but Haiku writes scratchpads about twice as long (2,350 vs. 1,050 characters on average), which adds to its CoT cost and latency on top of its higher token price.
8. **The fraud categories are simplified for teaching.** In the payments industry "carding" often covers all card testing, with BIN attacks as one kind; here they are separate labels, and carding is defined by the declines it produces even though the attacker is after approvals. The stolen-card examples on older accounts look more like account takeover; a stolen card more often arrives on a new or guest account. A synthetic identity usually checks out with a genuinely issued card, and merchants rarely see a "synthetic identity score". Real BIN attacks also return a mix of response codes, not only N7. The categories are built to have clear signals, which real fraud often does not.

## Shifted stress test and few-shot + CoT

Added in v1.1.0 to answer two open questions: does the rule engine only win because it knows the generator, and does chain-of-thought hurt by itself?

**Few-shot + CoT** ([`classify_llm_few-shot-cot.py`](classify_llm_few-shot-cot.py)) is the few-shot prompt (definitions and both examples) followed by the CoT checklist and output tags, so the only difference from few-shot is the step-by-step reasoning. On the original 50:

| Strategy | GPT-4o-mini | Claude Haiku 4.5 |
|---|---|---|
| Few-shot + CoT | 90% · 2,512 ms · $0.0112 | 90% · 19,126 ms · $0.2047 |

With the definitions back, reasoning costs accuracy only on Haiku (92% to 90%, one transaction) while multiplying latency and cost. Haiku's average is inflated by one request that took 609 s; its median is 7,305 ms. The client library gives up on a request after 10 minutes and retries, so that request most likely hung once and then succeeded on the retry, which is the only call the logged cost covers.

**Shifted set.** `python3 dataset_generator.py --shifted` builds a second 5,000-transaction set ([`shifted_fraud_dataset.json`](shifted_fraud_dataset.json), seed 7) where 45% of records get one category-specific set of changes that moves them just past a threshold of the rule engine. The usual 12% noise applies only to the records left unshifted (about 7% of the set). The rules were then left untouched. The 50-transaction LLM sample ([`shifted_sample_ids.txt`](shifted_sample_ids.txt)) takes 10 shifted records per category.

| Strategy | GPT-4o-mini | Claude Haiku 4.5 |
|---|---|---|
| Rule engine (frozen) | 0/50 = 0% · <0.01 ms · $0 | |
| Few-shot (shifted) | 62% · 602 ms · $0.0040 | 66% · 670 ms · $0.0320 |
| Few-shot + CoT (shifted) | 52% · 2,208 ms · $0.0118 | 84% · 8,248 ms · $0.2260 |

Correct out of 10, by category:

| Shifted category: what changed | Rules | Few-shot GPT | Few-shot Haiku | FS + CoT GPT | FS + CoT Haiku |
|---|---|---|---|---|---|
| Legitimate: gift order from an established account, shipped to another ZIP | 0 | 0 | 0 | 1 | 6 |
| BIN testing: bot throttled to 6-10 cards a minute | 0 | 10 | 3 | 10 | 6 |
| Carding: a working card from the list, charge approved | 0 | 1 | 10 | 0 | 10 |
| Stolen card: takeover of a 15-30 day old account | 0 | 10 | 10 | 5 | 10 |
| Synthetic identity: mainstream email, score 0.70-0.80 | 0 | 10 | 10 | 10 | 10 |

For the Legitimate row, 0 means every one of those real customers was flagged as fraud.

What this shows:

- **The rules' 94-99.5% does not transfer.** Each shift was built against a specific rule threshold, so 0% is by construction. The point is the contrast: the LLMs never saw those thresholds, and with few-shot each model gets three of the five shifted categories fully right, but not the same three (GPT-4o-mini misses approved carding, Claude Haiku 4.5 misses throttled BIN testing). On the full shifted set the rules get 2,710 of 2,726 unshifted records right and 0 of 2,274 shifted ones ([`shifted_rule_engine_eval_log.csv`](shifted_rule_engine_eval_log.csv)).
- **The narrative text steers the models, and it has thresholds of its own.** The LLMs read text written by `enrich_for_llm`, which is another hand-written rule layer. A synthetic score above 0.70 on a new account is described as "critically high", so the 10/10 on the synthetic row measures that sentence more than the model. Throttled bots run 2-8 s sessions, so their narrative says "Standard human behavior" (while still naming programmatic input). Whenever billing and shipping ZIPs differ, the narrative says "consistent with drop-houses", so three of the four setups flag 9-10 of the 10 gift orders as Stolen_Card. Throttled BIN testing reads "rotated through N different cards", the exact wording of the Carding example in the prompt, and few-shot Haiku follows the example.
- **Reasoning helped Haiku and hurt GPT-4o-mini on the unfamiliar cases**, by 5-9 transactions, which is close to the noise level for 50 transactions. Few-shot + CoT on Haiku gets 6 of 10 gift orders right (no other setup gets more than 1) and keeps every approved carding attempt; on GPT-4o-mini it drops young-account takeovers from 10/10 to 5/10.

This set is a stress test aimed at the rules, not a realistic traffic mix: 45% of legitimate orders become gift orders, for example, so the rules' overall accuracy on it (54%) means little. Use per-category results.

## How it works

```
dataset_generator.py ──> synthetic_fraud_dataset.json / .csv   (5,000 transactions, 92% legitimate)
                               │
       ┌───────────────────────┼─────────────────────────────┐
       │                       │                             │
classify_llm_zero-shot.py  classify_llm_few-shot.py  classify_llm_chain-of-thought.py
       │      (each run on OpenAI and Anthropic, same 50 IDs)  │
       └───────────────> llm_eval_log_<strategy>_<engine>.csv <┘
                               │
       ┌───────────────────────┼─────────────────────────┐
build_confusion_matrix.py  cost-accuracy_analysis.py    card_fraud_analyzer.py (rule baseline)
       │                       │                             │
confusion_matrix_analysis.json  performance_grid_analysis.json   rule_engine_eval_log.csv
                               │
                     generate_reports.py ──> VP_Report.md, Engineer_Report.md
```

1. `dataset_generator.py` makes five categories with a realistic imbalance (92% legitimate, 2% each for BIN testing, carding, stolen card, synthetic identity). 12% of records get 2-3 fields changed to create hard edge cases (a legitimate gift shipped to another ZIP, a synthetic account aged for months). Each record is then turned into the six sentences the LLM reads.
2. Each `classify_llm_*` script sends one prompt per transaction and logs the prediction, time, tokens and cost.
3. `build_confusion_matrix.py`, `cost-accuracy_analysis.py` and `card_fraud_analyzer.py` turn the logs into confusion matrices, the accuracy/time/cost grid and the rule-based baseline.
4. `generate_reports.py` writes both reports from those results, so their numbers always match the data.

## Run it

Python 3.10+.

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Put your OpenAI and Anthropic keys in `.env`. Run everything from the repo root. The committed dataset and logs let you skip straight to step 3.

```bash
python3 dataset_generator.py
```

Regenerates the dataset. The random seeds make the sampled values come out the same each time, but transaction IDs and timestamps change on every run, so `eval_sample_ids.txt` will no longer match. Skip this to reproduce the published results.

```bash
ENGINE=OPENAI python3 classify_llm_few-shot.py
```

Use `ENGINE=ANTHROPIC` for Claude, and the `zero-shot` / `chain-of-thought` / `few-shot-cot` scripts for the other strategies. The 8 runs on the original sample cost about $0.45 in total (most of it Haiku chain-of-thought).

```bash
python3 dataset_generator.py --shifted && python3 card_fraud_analyzer.py shifted_fraud_dataset.json shifted_rule_engine_eval_log.csv
```

Builds the shifted set and scores the rules on it. Transaction IDs are uuid4, so regenerating replaces `shifted_sample_ids.txt` and the committed `_shifted` logs no longer match it. Then add `DATASET=shifted` to the few-shot and few-shot-cot runs (all four shifted runs plus the two few-shot + CoT runs on the original sample cost about $0.49).

```bash
python3 build_confusion_matrix.py && python3 cost-accuracy_analysis.py && python3 card_fraud_analyzer.py && python3 generate_reports.py
```

## Files

| Path | What it is |
|---|---|
| `playground.ipynb` | The browser notebook described in [Start here](#play-with-it-in-your-browser) |
| `dataset_generator.py` | Builds the synthetic dataset and the LLM narratives |
| `synthetic_fraud_dataset.json` / `.csv` | The 5,000 transactions (nested for the LLM scripts, flat for tabular ML) |
| `eval_sample_ids.txt` | The 50 transaction IDs every LLM run uses |
| `classify_llm_*.py` | One script per prompting strategy (`few-shot-cot` added in v1.1.0) |
| `llm_eval_log_*.csv` | Per-transaction results of every run, including the full prompt or CoT output; `_shifted` files use the shifted sample |
| `shifted_fraud_dataset.json`, `shifted_sample_ids.txt` | Shifted stress-test set (each record has a `variant` field: `base`, `noisy` or `shifted`) and its 50-transaction LLM sample |
| `rule_engine_eval_log.csv`, `shifted_rule_engine_eval_log.csv` | Rule-engine predictions on each dataset |
| `build_confusion_matrix.py` | Confusion matrices, error asymmetries and never-confused pairs |
| `cost-accuracy_analysis.py` | Accuracy / time / cost grid |
| `card_fraud_analyzer.py` | Hand-written rule engine (the baseline) |
| `generate_reports.py` | Writes `VP_Report.md` and `Engineer_Report.md` from the data |
| `archive/` | First few-shot runs (Sep 2026), not used in the results. OpenAI: the published few-shot setup on the same 50 IDs (the shared sample was drawn from this run), identical predictions, slower latencies. Anthropic: a different random sample at the default temperature |
| `tests/`, `verify.sh` | Checks that the analysis code and reports are correct (below) |

## Verification

`./verify.sh` runs offline in about half a minute and makes no API calls. It checks that:

- no API key pattern or personal data is in the repo or anywhere in its history, and the current keys appear in no commit;
- the rule engine, matrix and grid code give the answers worked out by hand;
- every model run used its shared sample;
- every number in this README and the reports matches the logs;
- the playground notebook runs and its saved results are current;
- past bugs, put back in on purpose, are caught.

The checklist is in [`CLAUDE.md`](CLAUDE.md).

## About

By Ricardo Iguaran at [Fortuenti Consulting](https://fortuenti.co), to go with a LinkedIn article on testing AI for fraud decisions.

We help teams put AI to work and check that it does what they think it does, through consultancy and advisory or build projects, workshops for teams, 1:1 coaching and team coaching programs. If you want your team to run an experiment like this one on its own problem, write to [ricardo@fortuenti.co](mailto:ricardo@fortuenti.co).

## License

[MIT](LICENSE)
