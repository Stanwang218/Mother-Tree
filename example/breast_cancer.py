import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from prediction.mothertree import MotherTree
from prediction.gradTree import GradTreeClassifier
from huggingface_hub import hf_hub_download



def main():
    path = hf_hub_download(repo_id="ZiyuanStan/MotherTree",filename="mothertree.cpkt")   

    model = GradTreeClassifier(mothernet_path = path, init_from_mothernet=True, grid_search = True, grid_search_workers = 4)

    # Load sample data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # MotherTree performance
    model.fit(X_train, y_train)
    # Predict probabilities
    prediction_probabilities = model.predict_proba(X_test)
    # Predict labels
    predictions = model.predict(X_test)
    print("MotherTree-Tune Accuracy", accuracy_score(y_test, predictions))

    model = DecisionTreeClassifier(max_depth=5, random_state=42)  # For testing purposes, replace with actual model if needed
    # print(model)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("DecisionTreeClassifier Accuracy", accuracy_score(y_test, predictions))

    model = MotherTree(path=path, device="cpu", scale=True, inference_device="cpu", draw_tree=False,)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("MotherTree Accuracy", accuracy_score(y_test, predictions))
    


if __name__ == '__main__':
    main()