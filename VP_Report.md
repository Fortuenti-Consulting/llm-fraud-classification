# LLM fraud classification: summary for decision makers

## What was tested
Two language models, GPT-4o-mini and Claude Haiku 4.5, labeled 50 made-up card payments (10 per category, the same 50 for every setup) using three prompt styles. Two questions: can they sort a short text description of a payment into 5 categories, and are they fast enough to block a payment while it is being made? All data is synthetic; read the caveats in the README before applying any of these numbers to real traffic.

## Findings
1. **Too slow to block payments in real time.** The fastest setup (GPT-4o-mini, few-shot) averaged 531ms per transaction. A decision to block a card payment usually has to come back in under 200ms.
2. **Two examples closed most of the gap.** Adding 2 worked examples to the prompt (few-shot) raised accuracy from 74% to 90% on GPT-4o-mini and from 74% to 92% on Claude Haiku 4.5. Best result: 92% (Claude Haiku 4.5). A short list of hand-written rules got 47 of the same 50 right in well under a millisecond.
3. **Asking the model to reason step by step did not pay off on this sample.** Latency rose to as much as 6458ms, cost multiplied, and accuracy fell below few-shot (56% on GPT-4o-mini, 74% on Claude Haiku 4.5). The written reasoning shows the models dwelling on secondary signals. One caveat: that prompt listed only the category names, without the definitions or examples. With them added (few-shot + chain-of-thought), accuracy returns to 90% on GPT-4o-mini and 90% on Claude Haiku 4.5, so most of the drop came from the missing definitions. The reasoning step itself added time and cost without improving accuracy here.
4. **When the fraud changed, the models held up better than the rules.** On 50 payments built to slip past the rules, the rules got 0 right, as designed. The four few-shot setups scored 52-84%, best with Claude Haiku 4.5 reasoning step by step (84%). Rules are fast and cheap but only catch what someone already thought of.

## What to do next
1. Keep LLMs out of the real-time blocking path.
2. Turn the signals this pilot surfaced (card checks, velocity, account age, shipping address) into features for a conventional model such as XGBoost or a random forest, trained and tested on real labeled transactions. Such a model can answer in under 200ms.
3. Before trusting any model or rule set, test it on cases built to break it, the way the shifted set does here.
4. Possible side use: run the few-shot prompt after the fact, as a "shadow" pipeline that writes a plain-language explanation for the fraud review team once a transaction has been blocked.
