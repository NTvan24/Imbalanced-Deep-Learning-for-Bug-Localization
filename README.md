# Bug Localization using DNN

This project focuses on using Deep Neural Networks (DNN) to localize bugs in source code, aiming to identify the most relevant code elements (e.g., files, methods, or lines) associated with a given bug report.

## 🚀 Overview

Bug localization is a critical step in software debugging, aiming to reduce the time developers spend identifying the faulty code. In this project, we apply a DNN-based approach to learn the relationship between bug reports and code artifacts.

## 🧠 Model

We use a deep neural network trained on pairs of bug reports and source code files. The network learns semantic similarity and predicts the likelihood of a file being buggy.

Key components:
- Text preprocessing (TF-IDF, embeddings)
- Feature extraction from bug reports and code
- DNN with fully connected layers
- Binary classification or ranking

## 📈 Evaluation
We use metrics such as Top-k Accuracy, MAP, and MRR to evaluate localization performance.

## 📄 Reference Paper

This implementation is based on the approach proposed in the APSEC 2021 paper, which addresses the challenges of **data imbalance** in bug localization tasks. The paper introduces a deep learning model that utilizes **feature extraction**, **data balancing techniques**, and **ranking mechanisms** to improve localization performance, particularly on imbalanced datasets.

> Citation:
> > Bui, T. M. A., & Nguyen, V. L. (2021). An Imbalanced Deep Learning Model for Bug Localization. *APSEC 2021*.  
> > [Link to paper](https://ieeexplore.ieee.org/document/9719906)

