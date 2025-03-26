import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

# Step 1: Generate synthetic data (to get ~94% accuracy initially)
X, y = make_classification(n_samples=150, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42, flip_y=0.01)

# Step 2: Train a clean SVM classifier
clf = SVC(kernel='linear', C=0.05)
clf.fit(X, y)

# ✅ Plot Original Decision Boundary ===========================
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.title(f'SVM Decision Boundary (Before Poisoning)\nAccuracy: {accuracy_score(y, clf.predict(X)):.2%}')
plt.show()

# Step 3: Simulate a STRONG poisoning attack to reduce accuracy to ~60%
num_poison_samples = 80
X_poison = np.random.uniform(X.min(), X.max(), (num_poison_samples, 2))
y_poison = 1 - clf.predict(X_poison)  # Flip labels to maximize disruption

# Add poisoned samples to the training set
X_poisoned = np.vstack([X, X_poison])
y_poisoned = np.hstack([y, y_poison])

# Step 4: Retrain SVM on poisoned data
clf_poisoned = SVC(kernel='linear', C=0.05)
clf_poisoned.fit(X_poisoned, y_poisoned)

# ✅ Plot Decision Boundary After Poisoning ===========================
plt.figure(figsize=(8, 6))
plt.scatter(X_poisoned[:, 0], X_poisoned[:, 1], c=y_poisoned, cmap='coolwarm', edgecolors='k')
plt.scatter(X_poison[:, 0], X_poison[:, 1], c='yellow', edgecolors='black', label="Poisoned Samples")

Z = clf_poisoned.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.title(f'SVM Decision Boundary (After Strong Poisoning)\nAccuracy: {accuracy_score(y, clf_poisoned.predict(X)):.2%}')
plt.legend()
plt.show()

# Step 5: Adversarial Training (Recover to ~75% accuracy) ===========================
# Introduce moderate noise during adversarial training
X_adv = X_poisoned + np.random.normal(scale=0.3, size=X_poisoned.shape)
y_adv = y_poisoned

clf_adv = SVC(kernel='linear', C=0.05)
clf_adv.fit(X_adv, y_adv)

# ✅ Plot Decision Boundary After Adversarial Training ===========================
plt.figure(figsize=(8, 6))
plt.scatter(X_adv[:, 0], X_adv[:, 1], c=y_adv, cmap='coolwarm', edgecolors='k')

Z = clf_adv.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.title(f'SVM Decision Boundary (After Adversarial Training)\nAccuracy: {accuracy_score(y, clf_adv.predict(X)):.2%}')
plt.show()

# Step 6: Robust Learning using Isolation Forest (Recover to ~91%) ===========================
detector = IsolationForest(contamination=0.25, random_state=42)  # Adjust contamination level
inliers = detector.fit_predict(X_poisoned) == 1

X_filtered = X_poisoned[inliers]
y_filtered = y_poisoned[inliers]

clf_robust = SVC(kernel='linear', C=0.05)
clf_robust.fit(X_filtered, y_filtered)

# ✅ Plot Decision Boundary After Robust Learning ===========================
plt.figure(figsize=(8, 6))
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_filtered, cmap='coolwarm', edgecolors='k')

Z = clf_robust.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.title(f'SVM Decision Boundary (After Robust Learning)\nAccuracy: {accuracy_score(y, clf_robust.predict(X)):.2%}')
plt.show()

# ✅ Step 7: Print Accuracy Scores ================================
print(f"✅ Accuracy before poisoning: {accuracy_score(y, clf.predict(X)):.2%}")
print(f"❌ Accuracy after strong poisoning: {accuracy_score(y, clf_poisoned.predict(X)):.2%}")
print(f"🔒 Accuracy after adversarial training: {accuracy_score(y, clf_adv.predict(X)):.2%}")
print(f"🛡️ Accuracy after robust learning: {accuracy_score(y, clf_robust.predict(X)):.2%}")
