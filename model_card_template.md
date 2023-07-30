# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Task Description: Income Classification

The task at hand involves a classification problem aimed at predicting whether a person earns more than $50,000 per year. The dataset contains various features such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country. The goal is to build a machine learning model capable of accurately classifying individuals into two income groups: those earning more than $50,000 and those earning $50,000 or less.

## Intended Use
The developed model is designed for the specific task of binary classification, where the objective is to predict whether an individual earns more than $50,000 per year. At present, the model utilizes a predefined set of attributes/features to make these predictions effectively. However, it is important to note that the model's flexibility allows for future modifications, enabling the inclusion or exclusion of additional features as needed.

## Training Data
The dataset was obtained from the uci public repository and its extraction was done by Barry Becker from the 1994 Census database. The dataset has 32561 rows and 15 columns, out of which 8 are categorical, 6 numerical and 1 the target (>50k or <=50k). More information can be found (here)[https://archive.ics.uci.edu/ml/datasets/census+income]


# Model Metrics

Precision: 0.7167207792207793, Recall: 0.5631377551020408, fbeta: 0.6307142857142857
## Evaluation Data
For categorical features in the dataset, we performed categorical encoding using the same encoders that were utilized during the training phase. Specifically, we employed the Label Binarizer for the target variable and the One Hot Encoder for the categorical features. This ensured consistency and prevented any data leakage between the training and testing phase. 20% of the dataset was used for evaluation.

## Ethical Considerations

When working with sensitive variables such as sex, race, and workclass, it is crucial to handle them with special considerations to ensure the model's fairness and unbiased representation of the population.
## Caveats and Recommendations
The database is outdated.