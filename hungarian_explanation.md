### Hungarian Explanation

The Hungarian algorithm solves the assignment problem: Given n workers and n jobs, with a cost for each worker-job pair, find the one-to-one pairing with the lowest total cost.

In the tracker:
* the “workers” are the fly IDs from the previous frame and the “jobs” are the detections in the current frame. 
* The cost is usually the distance between where an ID was (or was predicted to be, if you use velocity) and where each detection is. 
* The algorithm picks the assignment that minimizes total distance across all flies at once.

##### So, you ask, why not just match each fly to its nearest detection?

Greedy matching fails when flies are close together. Say fly 1’s nearest detection is X, but fly 2’s only good option is also X. Greedy gives X to whoever is processed first, and the other fly gets a bad match. Hungarian looks at the whole picture and may give 1 a slightly worse match so that 2 gets a much better one, minimizing the sum. That’s what reduces ID swaps.

How it works:

1. Row reduction. Subtract each row’s minimum from every entry in that row. Every row now has at least one zero.
2. Column reduction. Subtract each column’s minimum from every entry in that column. Every column now has at least one zero.
3. Cover all zeros using the minimum number of horizontal and vertical lines.
4. Check for optimality. If the number of lines equals n, an optimal assignment exists among the zeros. Pick one zero per row and column and you’re done.
5. Otherwise, adjust. Find the smallest uncovered value. Subtract it from all uncovered entries and add it to entries covered by two lines. This creates new zeros without changing which assignment is optimal. Go back to step 3.