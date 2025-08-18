"""
Here you can visualize how this method works
"""
import numpy as np
import matplotlib.pyplot as plt
from ootcv import OutOfTimeSplit

# Generate toy data
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, 2, size=n_samples)
# if the data is already ordered in time, you can specify the number of periods. Otherwise, you should provide a list or
# an array the length of the X with each element showing which period it belongs to.
periods = 20
# Instantiate the splitter
cv = OutOfTimeSplit(n_splits=5, method='msa', forgetting=True)

# Visualize splits
plt.figure(figsize=(10, 4))
for fold_index, (train_idx, val_idx) in enumerate(cv.split(X, y, periods)):
    plt.scatter(train_idx, [fold_index]*len(train_idx), marker='o', label='Train' if fold_index==0 else "")
    plt.scatter(val_idx, [fold_index]*len(val_idx), marker='x', label='Validation' if fold_index==0 else "")

plt.xlabel("Sample Index")
plt.ylabel("Fold")
plt.xticks(np.arange(0, n_samples+1, 10))
plt.yticks(range(cv.n_splits), [f"Fold {i+1}" for i in range(cv.n_splits)])
plt.title("Out-of-Time CV: Training (o) vs Validation (x) Assignments")
plt.legend()
plt.tight_layout()
plt.show()
