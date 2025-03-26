# 🛡️ Robust ML Defense Demo: Data Poisoning + Mitigation

This repository contains a **Streamlit web app** that visually demonstrates a data poisoning attack against a machine learning model — and how to mitigate it using **robust learning techniques** like **Isolation Forests**.

---

## 🎯 Project Goals

- **Raise awareness** of adversarial ML threats (e.g. data poisoning)
- **Demonstrate visually** how an attack alters decision boundaries
- **Showcase defense** mechanisms to regain accuracy
- Provide a **business-friendly explainer** for leadership teams

---

## 🧠 What Is Data Poisoning?

Data poisoning is when an attacker injects malicious, mislabeled data into your training pipeline. This can:

- Corrupt model predictions
- Alter decision boundaries
- Create backdoors or classification errors

> 👨‍💻 How an attacker can poison data:

Step 1: Identify the Target
The attacker knows you use machine learning to approve loans.

They know (or guess) that your model is trained on past application data.

Step 2: Inject Poisoned Samples
They submit hundreds or thousands of loan applications like:

Fake applicants

Slightly modified real-looking data

Data that’s technically "clean" but purposefully crafted to trick the model

Example:

They intentionally lie on applications but use combinations that the system wrongly accepts.

These applications are approved and end up in your training logs.

Step 3: Wait for the Model to Retrain
Your system automatically retrains every month or quarter.

Those poisoned applications become part of the new training data.

✅ Result:
The AI learns to accept fraud as normal.

Next time, the attacker submits a high-value loan request — and the model approves it.

---

## ⚔️ The Attack Demo

1. We train a **linear SVM** on clean 2D synthetic data (~97% accuracy)
2. Inject 80 malicious samples with flipped labels (accuracy drops to ~47%)
3. Defend with **Isolation Forest** to filter poisoned points
4. Retrain for recovery (~92% accuracy)

---

## 🖥️ Streamlit App Features

- 4 side-by-side stages:
  - Clean model training
  - After poisoning
  - After robust learning


---


## 🚀 How to Run This App Locally

```bash
# Clone this repo
https://github.com/YOUR-ORG/YOUR-REPO.git
cd YOUR-REPO

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run app.py
```


---

## 🏢 How We Help

DevSecAI can:

- ✅ Audit your ML pipeline for poisoning vulnerabilities
- ✅ Integrate robust detection like Isolation Forests
- ✅ Design parallel training pipelines for safe model updates
- ✅ Deploy SecML to simulate and harden against adversarial attacks

> We help you move from reactive defense to proactive AI robustness.

---

## 🧩 About SecML

[SecML](https://github.com/pralab/secml) is a Python library for:

- Simulating **adversarial attacks** (evasion, poisoning, etc.)
- Running **security evaluations** on classifiers
- Supporting adversarial training workflows

It allows you to test your ML models in a red-team environment before attackers do.

---

---

## 📎 License
MIT License

---


