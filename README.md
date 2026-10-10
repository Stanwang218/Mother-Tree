# MotherTree
Code for paper:  <a href="https://arxiv.org/abs/2610.10832">*MotherTree: Meta-learning on synthetic data improves decision tree training*</a>.

MotherTree a tabular hypernetwork that is meta learned from sythentic dataset and can output a tree given the context. The downstream predictor is based on a gradient-based axis-aligned  decision tree,  <a href="https://arxiv.org/abs/2305.03515">GradTree</a>, (`GradTreeClassifier`). 

Trees generated from MotherTree benefit from the sythetic prior and establishes strong performance on low-sample regime. Besides. it improves over from-scratch gradient-based learning and acts as a strong initializer for specific task tuning. 


## Environment

Python 3.10 is recommended.

```bash
conda create -n mothertree python=3.10 -y
conda activate mothertree
pip install -r requirements.txt
```


## Examples

```bash
python example/breast_cancer.py
```

`example/breast_cancer.py`

1. downloads the pretrained MotherNet checkpoint from the Hugging Face Hub,
2. fits `GradTreeClassifier(init_from_mothernet=True)` on the breast-cancer dataset:
   the tree parameters are initialized from MotherTree and then trained on that dataset(MotherTree-Tune). Grid search is enabled. 
3. prints its accuracy next to a plain scikit-learn `DecisionTreeClassifier` baseline.


### GradTree

```python
from prediction.gradTree import GradTreeClassifier

model = GradTreeClassifier(epochs=200, patience=50)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### MotherTree(with out further tuning)
```python
from prediction.mothertree import MotherTree
model = MotherTree(path=path, device="cpu", scale=True, inference_device="cpu", draw_tree=False,)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("MotherTree Accuracy", accuracy_score(y_test, predictions))

```