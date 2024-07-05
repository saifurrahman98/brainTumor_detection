import os 
import cv2
import numpy as np #Numerical computing library.
import matplotlib.pyplot as plt #Plotting library.
from sklearn.model_selection import train_test_split#scikit-learn library
from sklearn.metrics import accuracy_score
#PCA is a technique for dimensionality reduction.
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')

def load_data(base_path):
    X = []
    Y = []
    classes = {'no_tumor': 0, 'pituitary_tumor': 1}

    for cls in classes:
        pth = os.path.join(base_path, 'Training', cls)
        for j in os.listdir(pth):
            img = cv2.imread(os.path.join(pth, j), 0)
            img = cv2.resize(img, (200, 200))
            X.append(img)
            Y.append(classes[cls])

    X = np.array(X)
    Y = np.array(Y)
    X_updated = X.reshape(len(X), -1)

    return X_updated, Y

def preprocess_data(X, Y):
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=10, test_size=.20)
    xtrain = xtrain / 255
    xtest = xtest / 255

    return xtrain, xtest, ytrain, ytest


def apply_pca(xtrain, xtest, target_variance=0.98):
    pca = PCA(target_variance)
    pca_train = pca.fit_transform(xtrain)
    pca_test = pca.transform(xtest)

    return pca_train, pca_test

def train_and_evaluate(xtrain, xtest, ytrain, ytest):
    lg = LogisticRegression(C=0.1) 
    sv = SVC()

    lg.fit(xtrain, ytrain)
    sv.fit(xtrain, ytrain)

    print("Logistic Regression Training Score:", lg.score(xtrain, ytrain))
    print("Logistic Regression Testing Score:", lg.score(xtest, ytest))
    print("SVM Training Score:", sv.score(xtrain, ytrain))
    print("SVM Testing Score:", sv.score(xtest, ytest))

def visualize_predictions(base_path, model):
    dec = {0: 'No Tumor', 1: 'Positive Tumor'}
    plt.figure(figsize=(12, 8))
    
    test_path = os.path.join(base_path, 'testing', 'no_tumor')
    c = 1
    
    for i in os.listdir(test_path)[:9]:
        img = cv2.imread(os.path.join(test_path, i), 0)
        img1 = cv2.resize(img, (200, 200))
        img1 = img1.reshape(1, -1) / 255
        p = model.predict(img1)
        
        plt.subplot(3, 3, c)
        plt.title(dec[p[0]])
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        c += 1
    
    plt.show()
    

def main():
    base_path = ''

    # Load Data
    X, Y = load_data(base_path)

    # Preprocess Data
    xtrain, xtest, ytrain, ytest = preprocess_data(X, Y)

    # Apply PCA
    pca_train, pca_test = apply_pca(xtrain, xtest)

    # Train and Evaluate Models
    print("Using Original Data:")
    train_and_evaluate(xtrain, xtest, ytrain, ytest)

    print("\nUsing PCA-Transformed Data:")
    train_and_evaluate(pca_train, pca_test, ytrain, ytest)

    # Visualize Predictions
    model = SVC()  # Change the model as needed
    model.fit(xtrain, ytrain)
    visualize_predictions(base_path, model)

if __name__ == "__main__":
    main()
