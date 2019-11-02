## CS 360 Lab 6 - Ensemble Methods

Name: Lamiaa Dakir

userId: ldakir

Number of Late Days Using for this lab: 0

---

### Analysis

1. Based on your ROC curves, which method would you choose for this application?
Which threshold would you choose? Does your answer depend on how much training
time you have available?

Based on the ROC curves, I would choose AdaBoost because with more training it reaches the ultimate goal of FPR = 0 and TPR = 1 using a threshold of 0.5.
If less training time is available, I would choose Random Forest.

2. `T` can be thought of as a measure of model complexity. Do these methods seem
to suffer from overfitting as the model complexity increases? What is the
intuition behind your answer?

These methods don't seem to suffer from overfitting as the model complexity increases. The ROC curve shows that more training (increasing T) leads to a lower false positive rate and a higher true positive rate which is a proof that the model is doing better when testing the data rather than overfitting the training data.
---

### Lab Questionnaire

(None of your answers below will affect your grade; this is to help refine lab assignments in the future)

1. Approximately, how many hours did you take to complete this lab? (provide your answer as a single integer on the line below)
7

2. How difficult did you find this lab? (1-5, with 5 being very difficult and 1 being very easy)
3

3. Describe the biggest challenge you faced on this lab: this lab was very clear and fairly easy.
