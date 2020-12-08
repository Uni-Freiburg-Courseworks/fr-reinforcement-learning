## 1.2 Value Iteration

### (b) What are similarities and diﬀerences between Value Iteration and Policy Iteration? Compare the two methods.

Similarities:

1. Both produces greedy policy from state values in the end
2. Both uses bootstrap to calculate state values

Differences:

1. Policy iteration needs to calculate best policy explicitly in each iteration, but value iteration don't.
2. Value iteration runs almost 4x faster than policy iteration in this case. 
