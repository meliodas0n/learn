# Supervised Learning
- One of the most commonly used and successful types of machine learning.
- It is used whenver we want to predict a certain outcome from a given input and we have examples of input/output pairs.
- Our goal is to make accurate predictions for new, never-before-seen data.
- Supervised learning often requires human effort to build the training set, but afterward automates and often speeds up an otherwise laborious or infeasible task.

## Classification and Regression
- Two majors of Supervised Learning are Classification and Regression.
- Classification: To predict a label, which is a choice from a predefined list of possibilties.
  - Ex: Classifying of irises into one of the three species.
- Classification is sometimes separated into Binay and Multiclass.
  - Binary is when there are only two classes to classify.
    - Yes/No
    - Spam/Not Spam
    - Positive/Negative
  - Multicalss is when there are more than two classes to classify.
    - Iris classification
    - Language classification
- Regression is to predict a continous number or a float-point/real number.
  - Predicting a person income, based on age, eduction and where they live.
  - Predicting the yield of a corn farm given attributes such as previous yields, weather and number of employees working on the farm.
- An easy way to distinguish between classification and regression tasks is to ask whether there is some kind of continuity in the output.

## Generalization, Overfitting and Underfitting
- In supervised learning, if a model is able to make accurate predictions on the unseen data, we say it is able to generalize from the training set to the test set.
- We want a model that is able to generalize as accurately as possible.