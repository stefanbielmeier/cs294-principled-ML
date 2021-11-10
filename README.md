# cs294-principled-ML

## File Dictionary

### datacapacityreq.py
Given data, calculate the following metrics:
1. Input Dimensionality: Number of input columns
2. Number of Points: Number of input rows
3. Class Balance: Ratio of points
4. Equivalent Energy Clusters: **To verify** Information content of data
5. Binary Decisions/Sample: **To verify** Entropy -> definiton in class?
6. Number of Thresholds: From Prof's capacity progression heuristic 
7. MEC: Memory needed to memorize output
8. Estimated Capacity Needed: MEC * columns **<- did we ever learn about this formula?**
9. Max Capacity Needed: thresholds * columns + thresholds + 1 <- **did we ever learn about this formula?**
10. Max Capacity After Log2: **What does this mean?**