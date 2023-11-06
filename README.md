# AMEX-Credit-Default-Prediction

Credit default prediction is central to managing risk in a consumer lending business. Credit default prediction allows lenders to optimize lending decisions, which leads to a better customer experience and sound business economics. Current models exist to help manage risk. But it's possible to create better models that can outperform those currently in use.

In this notebook, I will use machine learning to predict credit default. I will leverage an industrial scale data set to build a machine learning model that challenges the current model in production. Training, validation, and testing datasets include time-series behavioral data and anonymized customer profile information.

The dataset is from Kaggle:
https://www.kaggle.com/competitions/amex-default-prediction

The objective is to predict the probability that a customer does not pay back their credit card balance amount in the future based on their monthly customer profile. The target binary variable is calculated by observing 18 months performance window after the latest credit card statement, and if the customer does not pay due amount in 120 days after their latest statement date it is considered a default event.

The dataset contains aggregated profile features for each customer at each statement date. Features are anonymized and normalized, and fall into the following general categories:

D* = Delinquency variables S = Spend variables P_ = Payment variables B* = Balance variables R* = Risk variables with the following features being categorical:

['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

The goal is to predict, for each customer_ID, the probability of a future payment default (target = 1).

Note. The negative class has been subsampled for this dataset at 5%, and thus receives a 20x weighting in the scoring metric.

