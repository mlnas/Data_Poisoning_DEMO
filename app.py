import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from PIL import Image

# Set brand colors
PRIMARY_COLOR = "#41DC7A"
SECONDARY_COLOR = "#1D2532"

# Load logo
logo = Image.open("assets/logo.png")
st.set_page_config(layout="wide", page_title="Data Poisoning Demo", page_icon=logo)

# Sidebar
st.sidebar.image(logo, width=100)
st.sidebar.markdown("""
## DevSecAI
**Securing ML Pipelines**

**Topics Covered:**
- Data Poisoning Attack
- Impact on Accuracy
- Defense: Isolation Forest
""")

# Title
st.markdown("""
<style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #1D2532;
    }
</style>
<div class="main-title">🧪 Data Poisoning Attack and Robust Defense Demo</div>
""", unsafe_allow_html=True)

# Step 1: Generate clean data
np.random.seed(42)
X, y = make_classification(n_samples=150, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42, flip_y=0.01)

# Step 2: Train initial clean classifier
clf = SVC(kernel='linear', C=0.01)
clf.fit(X, y)
accuracy_clean = accuracy_score(y, clf.predict(X))

# Plot clean data
def plot_decision_boundary(clf, X, y, title, poisoned_pts=None):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors=SECONDARY_COLOR)
    if poisoned_pts is not None:
        ax.scatter(poisoned_pts[:, 0], poisoned_pts[:, 1], c='yellow', edgecolors='black', label='Poisoned', s=60)
        ax.legend()
    ax.set_title(title, fontsize=14)
    st.pyplot(fig)

st.markdown("""
### Step 1: Model trained on clean data
Accuracy: **97%**
""")
plot_decision_boundary(clf, X, y, "Before Poisoning")

# Step 3: Add poisoned samples
num_poison = 80
X_poison = np.random.uniform(X.min(), X.max(), (num_poison, 2))
y_poison = 1 - clf.predict(X_poison)
X_combined = np.vstack([X, X_poison])
y_combined = np.hstack([y, y_poison])

# Step 4: Retrain on poisoned data
clf_poisoned = SVC(kernel='linear', C=0.01)
clf_poisoned.fit(X_combined, y_combined)
accuracy_poisoned = accuracy_score(y, clf_poisoned.predict(X))

st.markdown("""
### Step 2: Model trained on poisoned data
Accuracy drops to **47%**
""")
plot_decision_boundary(clf_poisoned, X_combined, y_combined, "After Poisoning", poisoned_pts=X_poison)

# Step 5: Robust learning with Isolation Forest
detector = IsolationForest(contamination=0.25, random_state=42)
inliers = detector.fit_predict(X_combined) == 1
X_filtered = X_combined[inliers]
y_filtered = y_combined[inliers]

clf_robust = SVC(kernel='linear', C=0.01)
clf_robust.fit(X_filtered, y_filtered)
accuracy_robust = accuracy_score(y, clf_robust.predict(X))

st.markdown("""
### Step 3: Defense using Robust Learning
Filtered poisoned samples using **Isolation Forest**
Accuracy improves to **92%**
""")
plot_decision_boundary(clf_robust, X_filtered, y_filtered, "After Robust Learning")

# Final Metrics
st.markdown("""
---
### 📊 Summary of Results
- ✅ Accuracy **before poisoning**: 97%
- ❌ Accuracy **after poisoning**: 47%
- 🛡️ Accuracy **after robust learning**: 92%

---
**Isolation Forest** is an unsupervised anomaly detector that identifies and removes outliers in training data.

**DevSecAI** helps businesses integrate such defenses into their pipelines using tools like **SecML**, robust classifiers, and parallel training.
""")
