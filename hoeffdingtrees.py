import numpy as np
import joblib
from river import tree, metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import defaultdict

# === Load extracted features and labels ===
features = np.load('features.npy')  # Shape: (n_samples, n_features)
labels = np.load('labels.npy')      # Encoded integer labels
label_encoder = joblib.load('label_encoder.pkl')  # To convert labels to activity names

# === Initialize Hoeffding Tree (Online Decision Tree) and Accuracy Metric ===
model = tree.HoeffdingTreeClassifier()  # Online learning classifier
metric = metrics.Accuracy()             # Tracks accuracy over time

# === Track performance over time ===
accuracy_per_iteration = []   # Accuracy after each sample
true_labels = []              # For confusion matrix
predicted_labels = []         # For confusion matrix

# === Track per-class accuracy ===
class_correct = defaultdict(int)
class_total = defaultdict(int)

# === Online Training Simulation ===
print("🔁 Starting online training with Hoeffding Tree...\n")

for i in range(len(features)):
    # Format current sample as a dictionary for river
    x = {f'feat_{j}': val for j, val in enumerate(features[i])}
    y = labels[i]

    # Predict label before training
    y_pred = model.predict_one(x)

    # Evaluate prediction
    if y_pred is not None:
        metric.update(y, y_pred)  # Update accuracy
        class_total[y] += 1
        if y_pred == y:
            class_correct[y] += 1
    else:
        y_pred = -1  # Handle None predictions for debugging

    # Save predictions for confusion matrix
    true_labels.append(y)
    predicted_labels.append(y_pred)

    # Train the model on this sample (online learning)
    model.learn_one(x, y)

    # Record accuracy after this iteration
    accuracy_per_iteration.append(metric.get())

    # Status update
    if i % 100 == 0 and i != 0:
        print(f"[{i}] Accuracy so far: {metric.get():.4f}")

print("\n✅ Online training complete!")
print(f"🎯 Final Accuracy: {metric.get():.4f}")

# === Save Trained Model ===
joblib.dump(model, 'hoeffding_model.pkl')
print("💾 Model saved as 'hoeffding_model.pkl'")

# === Plot 1: Accuracy Over Time ===
plt.figure(figsize=(12, 6))
plt.plot(accuracy_per_iteration, label='Accuracy')
plt.xlabel('Iteration (Sample Index)')
plt.ylabel('Accuracy')
plt.title('Hoeffding Tree Accuracy Over Time')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Ensure All Classes Are Represented in Confusion Matrix ===
filtered_true = []
filtered_pred = []
for t, p in zip(true_labels, predicted_labels):
    if p != -1:  # Filter out None predictions (-1 is placeholder for debugging)
        filtered_true.append(t)
        filtered_pred.append(p)

# Ensure all classes are represented, even if unpredicted
labels_range = range(len(label_encoder.classes_))
cm = confusion_matrix(filtered_true, filtered_pred, labels=labels_range)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (After Online Training)')
plt.tight_layout()
plt.show()

# === Per-Class Accuracy ===
report = classification_report(filtered_true, filtered_pred, output_dict=True, target_names=label_encoder.classes_)

# Extract class-wise accuracy (recall = correctly predicted / total true)
class_names = list(report.keys())[:-3]  # Last 3 are avg/total/macro
class_accuracies = [report[name]['recall'] for name in class_names]

plt.figure(figsize=(10, 6))
sns.barplot(x=class_names, y=class_accuracies, palette='viridis')
plt.ylim(0, 1)
plt.xlabel('Activity Class')
plt.ylabel('Accuracy')
plt.title('Per-Class Accuracy (Hoeffding Tree)')
plt.tight_layout()
plt.show()

# === Print Tree Structure (Optional) ===
print("\n🌳 Hoeffding Tree Structure:")
print(model)
