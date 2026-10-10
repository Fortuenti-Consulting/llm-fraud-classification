import random
import uuid
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import json
from faker import Faker

# Seeds make categories, amounts and other sampled fields reproducible.
# Transaction/account/device IDs (uuid4) and timestamps (now) still change on every run.
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
NUM_SAMPLES = 5000

# Target distribution by category
CLASS_DISTRIBUTION = {
    "Legitimate": 0.92,
    "BIN_Testing": 0.02,
    "Carding": 0.02,
    "Stolen_Card": 0.02,
    "Synthetic_Identity": 0.02,
}


def sample_fraud_category():
    """Samples a category based on target class imbalance."""
    categories = list(CLASS_DISTRIBUTION.keys())
    probabilities = list(CLASS_DISTRIBUTION.values())
    return np.random.choice(categories, p=probabilities)


def generate_raw_transaction(category: str) -> dict:
    """Generates the raw fields of one transaction in the given category."""

    # Binary label alongside the category string
    is_fraud = 0 if category == "Legitimate" else 1

    # Base record identifiers
    tx_id = f"tx_{uuid.uuid4().hex[:10]}"
    timestamp = (
        datetime.now(timezone.utc) - timedelta(seconds=random.randint(0, 86400))
    ).isoformat()
    currency = "USD"
    account_id = f"acc_{uuid.uuid4().hex[:8]}"

    # Shared default lists
    common_domains = [
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "icloud.com",
        "hotmail.com",
    ]
    disposable_domains = [
        "temp-mail.org",
        "guerrillamail.com",
        "10minutemail.com",
        "throwawaymail.com",
    ]

    # Category-specific raw signals

    if category == "Legitimate":
        # 80% under $150, 15% up to $500, 5% high-value up to $3500
        amount = round(random.choices(
            [random.uniform(1.00, 150.00), random.uniform(150.00, 500.00), random.uniform(500.00, 3500.00)],
            weights=[0.80, 0.15, 0.05]
        )[0], 2)
        tx_type = random.choice(["Auth-and-Capture", "Auth-and-Capture", "Auth-Only"])
        mcc = random.choice([5411, 5812, 5912, 5311, 5651])  # Grocery, dining, pharmacy, department store, clothing
        bin_num = str(random.randint(400000, 499999))  # Standard Visa range
        last_4 = str(random.randint(1000, 9999))
        card_type = random.choice(["Credit", "Debit"])
        cvv_result = "Match"
        avs_result = random.choice(["Full Match", "Full Match", "Zip Match"])
        auth_code = "00 - Approved"

        account_age_days = random.randint(30, 1200)
        email_domain = random.choice(common_domains)
        email_domain_age_days = random.randint(365, 5000)

        zip_code = fake.zipcode()
        billing_zip = zip_code
        shipping_zip = zip_code  # Billing and shipping match

        synth_score = round(random.uniform(0.01, 0.20), 3)
        device_id = f"dev_{uuid.uuid4().hex[:8]}"
        ip_addr = fake.ipv4_public()
        user_agent = fake.user_agent()
        proxy_vpn = False if random.random() > 0.05 else True

        session_duration_ms = random.randint(12000, 120000)  # 12s to 2 mins
        input_method = random.choice(["Manual Typing", "Autofill"])
        pan_velocity_1m = random.randint(1, 2)
        cvv_fail_velocity_5m = 0

    elif category == "BIN_Testing":
        amount = random.choice([0.00, 1.00, 0.50, round(random.uniform(0.01, 1.50), 2)])
        tx_type = random.choice(["Pre-Auth", "Pre-Auth", "Auth-Only"])
        mcc = 5999  # Miscellaneous digital/online merchant
        bin_num = str(random.randint(400000, 599999))  # Visa/Mastercard range
        last_4 = str(random.randint(1000, 9999))
        card_type = "Credit"
        cvv_result = random.choice(["Mismatch", "Not Provided"])
        avs_result = "Mismatch"
        auth_code = "N7 - CVV Mismatch"

        account_age_days = 0  # Brand new guest session
        email_domain = random.choice(disposable_domains)
        email_domain_age_days = random.randint(1, 15)

        billing_zip = "00000"
        shipping_zip = "00000"

        synth_score = round(random.uniform(0.30, 0.60), 3)
        device_id = f"dev_bot_{uuid.uuid4().hex[:6]}"
        ip_addr = fake.ipv4_public()
        user_agent = "Python-urllib/3.10 / HeadlessChrome"
        proxy_vpn = True  # Datacenter / Proxy connection

        session_duration_ms = random.randint(150, 600)  # Sub-second bot execution
        input_method = "Programmatic Inject"
        pan_velocity_1m = random.randint(15, 60)  # High frequency
        cvv_fail_velocity_5m = random.randint(20, 80)

    elif category == "Carding":
        amount = round(random.uniform(0.50, 4.99), 2)  # Low micro-purchases
        tx_type = "Auth-Only"
        mcc = random.choice([5968, 5999, 7995])  # Subscriptions, misc. online, betting
        bin_num = str(random.choice([424242, 520082, 371449, 411111]))  # Well-known test-card BINs
        last_4 = str(random.randint(1000, 9999))
        card_type = random.choice(["Credit", "Debit", "Prepaid"])
        cvv_result = "Match"  # Stolen card lists ("fullz") include the real CVV
        avs_result = random.choice(["Zip Match", "Mismatch"])
        auth_code = random.choice(["51 - Insufficient Funds", "05 - Do Not Honor"])

        account_age_days = random.randint(0, 2)
        email_domain = random.choice(disposable_domains)
        email_domain_age_days = random.randint(1, 30)

        billing_zip = fake.zipcode()
        shipping_zip = fake.zipcode()

        synth_score = round(random.uniform(0.20, 0.50), 3)
        device_id = f"dev_script_{uuid.uuid4().hex[:6]}"
        ip_addr = fake.ipv4_public()
        user_agent = "Mozilla/5.0 (X11; Linux x86_64) Selenium/4.1.0"
        proxy_vpn = True  # Residential proxy network

        session_duration_ms = random.randint(500, 1500)  # Scripted paste action
        input_method = "Programmatic Inject"
        pan_velocity_1m = random.randint(5, 20)  # Rotating cards quickly
        cvv_fail_velocity_5m = random.randint(1, 5)

    elif category == "Stolen_Card":
        amount = round(random.uniform(180.00, 2200.00), 2)  # High cash-out value
        tx_type = "Auth-and-Capture"
        mcc = random.choice([5732, 5094, 5944])  # Electronics, Jewelry, High-value goods
        bin_num = str(random.randint(400000, 550000))
        last_4 = str(random.randint(1000, 9999))
        card_type = "Credit"
        cvv_result = "Match"
        avs_result = "Full Match"  # Genuine billing address used to pass AVS
        auth_code = "00 - Approved"  # Auth succeeds initially

        account_age_days = random.randint(180, 900)  # Account takeover on a mature user
        email_domain = random.choice(common_domains)
        email_domain_age_days = random.randint(365, 3000)

        billing_zip = fake.zipcode()
        shipping_zip = fake.zipcode()  # Independent random ZIP, so almost always a mismatch (drop-house)

        synth_score = round(random.uniform(0.05, 0.25), 3)
        device_id = f"dev_unrecognized_{uuid.uuid4().hex[:6]}"
        ip_addr = fake.ipv4_public()
        user_agent = fake.user_agent()
        proxy_vpn = True  # Fraudster hiding location

        session_duration_ms = random.randint(18000, 60000)  # Human session
        input_method = random.choice(["Manual Typing", "Autofill"])
        pan_velocity_1m = 1
        cvv_fail_velocity_5m = 0

    elif category == "Synthetic_Identity":
        amount = round(random.uniform(250.00, 1200.00), 2)
        tx_type = "Auth-and-Capture"
        mcc = random.choice([5541, 5311, 5732])  # Gas, department store, electronics
        bin_num = str(random.randint(400000, 550000))
        last_4 = str(random.randint(1000, 9999))
        card_type = "Credit"
        cvv_result = "Match"
        avs_result = "Full Match"
        auth_code = "00 - Approved"

        account_age_days = random.randint(1, 10)  # Recently registered account
        email_domain = random.choice(disposable_domains)
        email_domain_age_days = random.randint(2, 25)

        synth_zip = fake.zipcode()
        billing_zip = synth_zip
        shipping_zip = synth_zip

        synth_score = round(random.uniform(0.78, 0.98), 3)  # High synthetic alert score
        device_id = f"dev_synth_{uuid.uuid4().hex[:6]}"
        ip_addr = fake.ipv4_public()
        user_agent = fake.user_agent()
        proxy_vpn = False

        session_duration_ms = random.randint(25000, 90000)  # Human session
        input_method = "Manual Typing"
        pan_velocity_1m = 1
        cvv_fail_velocity_5m = 0

    return {
        "transaction_id": tx_id,
        "timestamp": timestamp,
        "amount": amount,
        "currency": currency,
        "transaction_type": tx_type,
        "mcc": mcc,
        "pan_token": f"tok_{bin_num}{last_4}",
        "bin": bin_num,
        "last_4": last_4,
        "card_type": card_type,
        "cvv_result": cvv_result,
        "avs_result": avs_result,
        "auth_response_code": auth_code,
        "account_id": account_id,
        "account_age_days": account_age_days,
        "email_domain": email_domain,
        "email_domain_age_days": email_domain_age_days,
        "billing_zip": billing_zip,
        "shipping_zip": shipping_zip,
        "identity_synthetic_score": synth_score,
        "device_fingerprint": device_id,
        "ip_address": ip_addr,
        "user_agent": user_agent,
        "proxy_vpn_flag": proxy_vpn,
        "session_duration_ms": session_duration_ms,
        "field_input_method": input_method,
        "velocity_pan_count_1m_ip": pan_velocity_1m,
        "velocity_cvv_fail_count_5m_bin": cvv_fail_velocity_5m,
        "is_fraud": is_fraud,
        "fraud_category": category,
    }


def generate_noisy_variant(raw: dict, category: str) -> dict:
    """
    Changes 2-3 fields of a generated transaction to create harder edge cases,
    so a classifier cannot rely on one clean signal per category.
    """
    common_domains = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "hotmail.com"]
    
    if category == "Legitimate":
        # Suspicious Legitimate (e.g., buying a gift on VPN, new account)
        raw["proxy_vpn_flag"] = True
        raw["shipping_zip"] = fake.zipcode()  # Intentional mismatch (gift shipping)
        raw["account_age_days"] = random.randint(0, 2)  # Brand new customer
        
    elif category == "BIN_Testing":
        # Bot trying to look human: real email domain, manual typing speed
        raw["email_domain"] = random.choice(common_domains)
        raw["session_duration_ms"] = random.randint(12000, 25000)
        raw["field_input_method"] = "Manual Typing"
        
    elif category == "Carding":
        # Carder pushing their luck with a higher amount and matching AVS
        raw["amount"] = round(random.uniform(15.00, 65.00), 2)
        raw["avs_result"] = "Full Match"
        raw["session_duration_ms"] = random.randint(15000, 45000)
        
    elif category == "Stolen_Card":
        # Testing the stolen card first with a low amount, or buying a digital good
        if random.random() > 0.5:
            raw["amount"] = round(random.uniform(1.50, 10.00), 2)  # Probe test
        else:
            raw["shipping_zip"] = raw["billing_zip"]  # Digital good (zips match)
        
        raw["proxy_vpn_flag"] = False  # Fraudster forgot to turn on VPN
        
    elif category == "Synthetic_Identity":
        # "Sleeper" synthetic account: aged for months, normal email, lower score
        raw["account_age_days"] = random.randint(180, 400)
        raw["email_domain"] = random.choice(common_domains)
        raw["identity_synthetic_score"] = round(random.uniform(0.40, 0.65), 3)
        
    return raw


def enrich_for_llm(raw: dict) -> dict:
    """
    Pass 2: deterministic enrichment.
    Translates the raw numerical/categorical values into short natural-language
    narratives that the LLM classifiers read instead of the raw fields.
    """
    
    # 1. Amount narrative
    if raw["amount"] <= 1.50 and raw["transaction_type"] == "Pre-Auth":
        amount_narrative = f"A {raw['currency']} {raw['amount']:.2f} pre-authorization request designed to validate the card without a full charge."
    elif raw["amount"] <= 5.00:
        amount_narrative = f"A very low-value micro-purchase of {raw['currency']} {raw['amount']:.2f}."
    else:
        amount_narrative = f"A standard order value of {raw['currency']} {raw['amount']:.2f} via {raw['transaction_type']}."

    # 2. Verification outcome narrative
    # Substring checks so both "51 - Insufficient Funds" and "05 - Do Not Honor" match
    if raw["cvv_result"] == "Mismatch" or "N7" in raw["auth_response_code"]:
        verification_story = f"Authorization failed due to a CVV Mismatch (Code {raw['auth_response_code']})."
    elif "51" in raw["auth_response_code"] or "05" in raw["auth_response_code"]:
        verification_story = f"CVV matched, but the transaction was declined by the issuer (Code {raw['auth_response_code']} - Insufficient Funds/Do Not Honor)."
    else:
        verification_story = "Authorization approved. CVV and billing address checks passed successfully."

    # 3. Velocity and card rotation narrative
    # Rolling time-window counts become a description of the attack pattern
    if raw["velocity_pan_count_1m_ip"] > 10 and raw["velocity_cvv_fail_count_5m_bin"] > 10:
        velocity_story = f"Extreme velocity anomaly: This IP has attempted {raw['velocity_pan_count_1m_ip']} distinct card numbers in the last 60 seconds, triggering massive CVV failures on BIN {raw['bin']}."
    elif raw["velocity_pan_count_1m_ip"] > 4:
        velocity_story = f"High velocity: This session has rotated through {raw['velocity_pan_count_1m_ip']} different cards in under a minute."
    else:
        velocity_story = "Normal velocity: Only 1-2 cards attempted from this IP recently."

    # 4. Identity and account age narrative
    # Accounts younger than 14 days are never described as "established"
    if raw["account_age_days"] == 0:
        account_story = "This is a brand-new guest checkout session with zero historical platform trust."
    elif raw["account_age_days"] < 14:
        if raw["identity_synthetic_score"] > 0.70:
            account_story = f"Account is highly suspicious. Created {raw['account_age_days']} days ago using a disposable email ({raw['email_domain']}). Synthetic identity score is critically high ({raw['identity_synthetic_score']})."
        else:
            account_story = f"Newly created account, active for only {raw['account_age_days']} days. Email domain is {raw['email_domain']}."
    else:
        account_story = f"Established user account, active for {raw['account_age_days']} days."

    # 5. Logistics narrative
    # Flags physical goods shipped to a ZIP other than the billing ZIP
    if raw["billing_zip"] == "00000" or raw["transaction_type"] == "Pre-Auth":
        address_story = "No physical delivery address provided (digital/pre-auth transaction)."
    elif raw["billing_zip"] != raw["shipping_zip"]:
        address_story = f"High logistics risk: Billing ZIP ({raw['billing_zip']}) matches the cardholder, but items are being shipped to a completely different ZIP ({raw['shipping_zip']}), consistent with drop-houses."
    else:
        address_story = "Low logistics risk: Billing and shipping ZIP codes match."

    # 6. Behavior and network narrative
    # Sub-second sessions with injected input read as a bot
    if raw["field_input_method"] == "Programmatic Inject" and raw["session_duration_ms"] < 2000:
        behavior_story = f"Severe bot signature: Checkout completed in an impossible {raw['session_duration_ms']} milliseconds. Fields were programmatically injected bypassing standard keyboard events."
    else:
        seconds = raw['session_duration_ms'] / 1000
        behavior_story = f"Standard human behavior: Checkout took {seconds:.1f} seconds using {raw['field_input_method']}."

    if raw["proxy_vpn_flag"]:
        network_story = "Connection originated from a datacenter or residential proxy IP, masking the true origin."
    else:
        network_story = "Connection originated from a standard residential/mobile ISP."

    # Final payload
    # raw_ml_features feeds tabular models (rule engine, gradient-boosted trees);
    # llm_reasoning_prompt is what the LLM classifiers see.
    return {
        "raw_ml_features": raw,
        "llm_reasoning_prompt": {
            "amount_and_intent": amount_narrative,
            "verification_outcome": verification_story,
            "velocity_context": velocity_story,
            "identity_profile": account_story,
            "logistics_context": address_story,
            "behavior_and_network": f"{behavior_story} {network_story}"
        }
    }


if __name__ == "__main__":
    print(f"Generating {NUM_SAMPLES} synthetic transactions. This may take a few seconds...")
    
    # Step 1: Sample transaction categories based on set weighted distribution
    sampled_types = [sample_fraud_category() for _ in range(NUM_SAMPLES)]

    # Step 2: Pass 1, raw fields plus noise
    raw_records = []
    noisy_count = 0
    
    for cat in sampled_types:
        # Generate the baseline transaction
        record = generate_raw_transaction(cat)
        
        # 12% of the time, perturb the fields to create a noisy edge-case
        if random.random() < 0.12:
            record = generate_noisy_variant(record, cat)
            noisy_count += 1
            
        raw_records.append(record)
        
    print(f"Injected noise into {noisy_count} records (~{(noisy_count/NUM_SAMPLES)*100:.1f}% of total).")

    # Step 3: Pass 2, narratives for the LLM
    enriched_records = [enrich_for_llm(record) for record in raw_records]

    # Print the final distribution
    counts = pd.Series(sampled_types).value_counts()
    print(f"\n--- FINAL {NUM_SAMPLES}-SAMPLE DISTRIBUTION ---")
    print(counts)

    # Export 1: nested JSON for the LLM scripts
    json_filename = "synthetic_fraud_dataset.json"
    with open(json_filename, "w") as f:
        json.dump(enriched_records, f, indent=2)
    print(f"\nSaved nested LLM format to {json_filename}")

    # Export 2: flat CSV for tabular ML (XGBoost/Random Forest)
    flat_records = []
    for record in enriched_records:
        # Copy the raw features
        flat_dict = record["raw_ml_features"].copy()
        
        # Narrative fields get an llm_ prefix
        for key, value in record["llm_reasoning_prompt"].items():
            flat_dict[f"llm_{key}"] = value
            
        flat_records.append(flat_dict)

    df_final = pd.DataFrame(flat_records)
    csv_filename = "synthetic_fraud_dataset.csv"
    df_final.to_csv(csv_filename, index=False)
    print(f"Saved flattened ML format to {csv_filename}")
