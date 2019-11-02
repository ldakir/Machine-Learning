## CS 360 Lab 1 - K-nearest neighbors

Name: Lamiaa Dakir

userId: ldakir

Number of Late Days Using for this lab: 0

---

### Analysis Questions

1. What values of k did you try?
-> I tried k from 1 to 10
2. Which value of k produced the highest accuracy? What general trends did you
observe as k increased?
-> The value of K that produced the highest accuracy is k = 1. In general, as K
increases, the accuracy decreases.
3. When using the entire training dataset, what are your observations about the
runtime of K-nearest neighbors? List 1-2 ideas for making this algorithm faster.
-> Using the entire training dataset takes a very long time to run just for k=1.
-> Ideas to make this algorithm faster:
  a- Calculate the distance between a test input and training inputs concurrently instead of simultaneously. This allows
  all the distances to be calculated at the same time instead of one after the other
  b- Another way is to record all the distances and not re-calculate them for every K.
  For example: for k = 1 we have one nearest neighbor. For k = 2, we use the nearest neighbor we found for k = 1 and only calculate
  the second nearest neighbor.
---

### Lab Questionnaire

(None of your answers below will affect your grade; this is to help refine lab
assignments in the future)

1. Approximately, how many hours did you take to complete this lab? (provide
  your answer as a single integer on the line below)
-> 5
2. How difficult did you find this lab? (1-5, with 5 being very difficult and 1
  being very easy)
-> 3
3. Describe the biggest challenge you faced on this lab:
-> The most challenging part of this lab was finding an optimal way to find the nearest neighbors
