# Decision-Trees-Features-Subset-Selection

In this project we want to evaluate the ability of multiple subset selecting algorithms in choosing best features to maximize decision tree performance, we
will be using 8 different algorithms and test them on 10 datasets.

# Algorithms:
Each algorithm takes Data that has `N` features, and parameter `p` that represents the percentage of how many features to take, and will return `k = p · N` features:
  1. **`Random :`** Returns a random subset of size k
  2. **`MaxIG :`** Calculates the Information Gain of each feature and returns the K gain with the highest IG value.
  3. **`Diverse1 :`** Initialize a window of size `min(2k, N)` with the highest IG value and then iteratively `k-1 times`: pop feature with the farther distance from other features, based on `Spearman Correlation` measure.
  4. **`Diverse2 :`** As diverse1 but in the opposite order: Initialize a window of size `min(2k, N)` with the farthest features distance from other features, based on `Spearman Correlation` measure. Then returns the K gain with the highest IG value from the window.
  5. **`WrapperForward :`** Start with the empty set and repeat K times: For any feature `f` not yet selected, select the feature with maximum accuracy when training `Current subset + f`. based on Cross-validation experiment.
  6. **`WrapperBackward :`** As before - but start with the set of all features, and pop the worst feature until you are left with K features.
  7. **`LocalSearch : `** first-choice hill climbing algorithm.
  8. **`randomForestFeatures`** Use the features that help us the most to teach the decision trees (the features that the trees use the most to split) we built a decision forest and for each feature, appearance in the tree is weighted by the relative number of examples that have passed through its node in the tree. we choose the best `k` features based on this weight.

# Methodology
For each dataset we split it into 5 splits, 4 of splits will be used as training data the 5th as test data, repeated 5 times where each time we choose different one as test data, then we procced to evaluate for each p and algorithm.

For readability, we will be calling each dataset as : (i)ds(j), where i is just to sort the datasets from 1-10, and j is the number of features of each dataset(not including label), datasets where p*N is less than 1 were set an accuracy of 0 in this current p.

# Conclusion
All algorithms performed better the more features they had, in general.
Diverse 2 algorithm seems to give decent results on p=0.2.
Diverse 1 algorithm seems to be the most consistent to use if we want to cut the number of features at our disposal, it’s also relatively fast compared to the other algorithms we tested.

# References

* “Introduction to Artificial Intelligence” course: https://webcourse.cs.technion.ac.il/236501
* datasets:
  - https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition 
  - https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition 
  - https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset) 
  - https://archive.ics.uci.edu/ml/datasets/ionosphere 
  - https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope 
  - https://archive.ics.uci.edu/ml/datasets/Ultrasonic+flowmeter+diagnostics 
  - https://archive.ics.uci.edu/ml/datasets/Parkinson+Dataset+with+replicated+acoustic+features+ 
  - https://archive.ics.uci.edu/ml/machine-learning-databases/00230/ https://archive.ics.uci.edu/ml/datasets/SPECTF+Heart 
  - https://archive.ics.uci.edu/ml/datasets/Quality+Assessment+of+Digital+Colposcopies 
  - https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes

* Spearman rank correlation coefficient:
https://mathworld.wolfram.com/SpearmanRankCorrelationCoefficient.html
* sklearn feature selection: https://scikit-learn.org/stable/modules/feature_selection.html
