import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

# Set the company's brand colors
primary_color = '#41DC7A'  # Green
secondary_color = '#1D2532'  # Dark Blue

def plot_graph(X, y, clf, title, accuracy):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors=secondary_color)
    plt.title(f'{title}\nAccuracy: {accuracy:.2%}')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.grid(True)
    st.pyplot(plt)

# Streamlit app main function
def main():
    st.title("Data Poisoning Attack and Defense Simulation")
    
    # Step 1: Generate synthetic data (to get ~97% accuracy initially)
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, random_state=42, flip_y=0.01)
    
    # Step 2: Train a clean SVM classifier
    clf = SVC(kernel='linear', C=0.05)
    clf.fit(X, y)

    # Plot Original Decision Boundary
    plot_graph(X, y, clf, 'SVM Decision Boundary (Before Poisoning)', accuracy_score(y, clf.predict(X)))

    # Step 3: Simulate a STRONG poisoning attack to reduce accuracy to ~47%
    num_poison_samples = 80
    X_poison = np.random.uniform(X.min(), X.max(), (num_poison_samples, 2))
    y_poison = 1 - clf.predict(X_poison)  # Flip labels to maximize disruption

    # Add poisoned samples to the training set
    X_poisoned = np.vstack([X, X_poison])
    y_poisoned = np.hstack([y, y_poison])

    # Step 4: Retrain SVM on poisoned data
    clf_poisoned = SVC(kernel='linear', C=0.05)
    clf_poisoned.fit(X_poisoned, y_poisoned)

    # Plot Decision Boundary After Poisoning
    plot_graph(X_poisoned, y_poisoned, clf_poisoned, 'SVM Decision Boundary (After Strong Poisoning)', accuracy_score(y, clf_poisoned.predict(X)))

    # Step 5: Robust Learning using Isolation Forest (Recover to ~92%)
    detector = IsolationForest(contamination=0.25, random_state=42)  # Adjust contamination level
    inliers = detector.fit_predict(X_poisoned) == 1

    X_filtered = X_poisoned[inliers]
    y_filtered = y_poisoned[inliers]

    clf_robust = SVC(kernel='linear', C=0.05)
    clf_robust.fit(X_filtered, y_filtered)

    # Plot Decision Boundary After Robust Learning
    plot_graph(X_filtered, y_filtered, clf_robust, 'SVM Decision Boundary (After Robust Learning)', accuracy_score(y, clf_robust.predict(X)))

    # Print Accuracy Scores
    st.write(f"✅ Accuracy before poisoning: {accuracy_score(y, clf.predict(X)):.2%}")
    st.write(f"❌ Accuracy after strong poisoning: {accuracy_score(y, clf_poisoned.predict(X)):.2%}")
    st.write(f"🛡️ Accuracy after robust learning: {accuracy_score(y, clf_robust.predict(X)):.2%}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
