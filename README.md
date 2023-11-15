# AOA-LAB
# 24
## CHERRY PICK UP PROBLEM
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/12043ed6-90da-4ffc-8f1d-a6b4670f0810)
```
class Solution(object):
    def cherryPickup(self, grid):
        rn = len(grid)
        cn = len(grid[0])
        dp=[[[0]*cn for _ in grid[0]] for _ in grid ]
        for i in range(cn):
            for j in range(cn):
                dp[-1][i][j]=grid[-1][i] if i==j else grid[-1][i]+grid[-1][j]
        for k in range(rn-2,-1,-1):
            r=grid[k]
            for i in range(cn):
                for j in range(i,cn):
                    for di in [-1,0,1]:
                        for dj in [-1,0,1]:
                            if 0 <= i+di < cn and 0 <= j+dj < cn:
                                if i==j:
                                    dp[k][i][j]=max(dp[k][i][j],dp[k+1][i+di][j+dj]+r[i])
                                else:
                                    dp[k][i][j]=max(dp[k][i][j],dp[k+1][i+di][j+dj]+r[i]+r[j])
                                        
                    
        return dp[0][0][cn - 1]
        
grid=[[3,1,1],
      [2,5,1],
      [1,5,5],
      [2,1,1]]
ob=Solution()
print(ob.cherryPickup(grid))
```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/382c784d-ffa4-436e-a610-bf2eab152f4c)
```
class Solution(object):
    def cherryPickup(self, grid):
        n=len(grid)
        dp=[[-1]*(n+1) for _ in range(n+1)]
        dp[1][1]=grid[0][0]
        for m in range(1,(n<<1)-1):
            for i in range(min(m,n-1),max(-1,m-n),-1):
                for p in range(i,max(-1,m-n),-1):
                    j,q=m-i,m-p
                    if grid[i][j]==-1 or grid[p][q]==-1:
                        dp[i+1][p+1]=-1
                    else:
                        dp[i+1][p+1]=max(dp[i+1][p+1],dp[i][p+1],dp[i+1][p],dp[i][p])
                        if dp[i+1][p+1]!=-1:dp[i+1][p+1]+=grid[i][j]+(grid[p][q] if i!=p else 0)
        return max(0,dp[-1][-1])
obj=Solution()
grid=[[0,1,-1],[1,0,-1],[1,1,1]]        
print(obj.cherryPickup(grid))
```
## KNAPSACK PROBLEM
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/b7a2ed94-1896-4849-9aa8-7cb58efabe32)
```
def knapSack(W, wt, val, n):
    if n==0 or W ==0:
        return 0
    if wt[n-1]>W:
        return knapSack(W, wt, val, n-1)
    else:
        return max(val[n-1]+knapSack(W-wt[n-1], wt, val, n-1),knapSack(W, wt, val, n-1))
############### Add your code here ##############

x=int(input())
y=int(input())
W=int(input())
val=[]
wt=[]
for i in range(x):
    val.append(int(input()))
for y in range(y):
    wt.append(int(input()))
n = len(val)
print('The maximum value that can be put in a knapsack of capacity W is: ',knapSack(W, wt, val, n))
```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/f53f94c2-83c3-4224-8c2c-06d2d2b1a680)
```
def knapSack(W, wt, val, n):
    ########## Add your code here #########
    if n==0 or W==0:
        return 0
    if wt[n-1]>W:
        return knapSack(W, wt, val, n-1)
    else:
        return max(val[n-1]+knapSack(W-wt[n-1], wt, val, n-1),knapSack(W, wt, val, n-1))

x=int(input())
y=int(input())
W=int(input())
val=[]
wt=[]
for i in range(x):
    val.append(int(input()))
for y in range(y):
    wt.append(int(input()))

n = len(val)
print('The maximum value that can be put in a knapsack of capacity W is: ',knapSack(W, wt, val, n))
```
## TRAVELLING SALES MAN PROBLEM
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/b52cce75-9929-4f21-92e6-d78ec23dd244)

```
from sys import maxsize
from itertools import permutations
V = 4
 

def travellingSalesmanProblem(graph, s):
	vertex = []
	for i in range(V):
		if i != s:
			vertex.append(i)
	min_path = maxsize
	next_permutation=permutations(vertex)
	for i in next_permutation:
		current_pathweight = 0

		# compute current path weight
		k = s
		for j in i:
			current_pathweight += graph[k][j]
			k = j
		current_pathweight += graph[k][s]
		min_path = min(min_path, current_pathweight)
		
	return min_path
if __name__ == "__main__":
 
    graph = [[0, 10, 15, 20], [10, 0, 35, 25],
            [15, 35, 0, 30], [20, 25, 30, 0]]
    s = 0
    print(travellingSalesmanProblem(graph, s))
```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/1d6f40a5-bc91-4249-8376-021e574ac84b)
```
from typing import DefaultDict
 
 
INT_MAX = 2147483647
 

def findMinRoute(tsp):
    sum = 0
    counter = 0
    j = 0
    i = 0
    min = INT_MAX
    visitedRouteList = DefaultDict(int)
 
    
    visitedRouteList[0] = 1
    route = [0] * len(tsp)
 
    
    while i < len(tsp) and j < len(tsp[i]):
        if counter >= len(tsp[i]) - 1:
            break
 
        # If this path is unvisited then
        # and if the cost is less then
        # update the cost
        if j != i and (visitedRouteList[j] == 0):
            if tsp[i][j] < min:
                min = tsp[i][j]
                route[counter] = j + 1
 
        j += 1
 
        # Check all paths from the
        # ith indexed city
        if j == len(tsp[i]):
            sum += min
            min = INT_MAX
            visitedRouteList[route[counter] - 1] = 1
            j = 0
            i = route[counter] - 1
            counter += 1
 
    # Update the ending city in array
    # from city which was last visited
    i = route[counter - 1] - 1
 
      
 
    for j in range(len(tsp)):
 
        if (i != j) and tsp[i][j] < min:
            min = tsp[i][j]
            route[counter] = j + 1
 
    sum += min
 
    
    print("Minimum Cost is :", sum)
 
 # Driver Code
if __name__ == "__main__":
 
  
    tsp = [[-1, 30, 25, 10], 
[15, -1, 20, 40], 
[10, 20, -1, 25], 
[30, 10, 20, -1]] 
 
    # Function Call
    findMinRoute(tsp)

```
## BRUTE FORCE ALGORITHM
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/335304b8-c243-444d-b32a-805aed7dc4d7)

```
def match(string,sub):
    l = len(string)
    ls = len(sub)
    for i in range(l-ls+1):
        if string[i:i + ls] ==sub:
            print(f"Found at index {i}")

    ########### Add your code here #######

str1=input()
str2=input()

```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/32886996-4919-4294-96ab-eb0dd7c1f449)
```
def find_maximum(lst):
    if not lst:
        return None
    mv=lst[0]
    for i in lst:
        if i>mv:
            mv=i
    return mv
#############  Add your code here ##############

test_scores = []
n=int(input())
for i in range(n):
    test_scores.append(int(input()))
print("Maximum value is ",find_maximum(test_scores))
```
# 23
## DAY -1 Minimum Cost Path
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/c5afd29f-61cd-4b04-9627-fb967d418da2)
```
R = int(input())
C = int(input())
def minCost(cost, m, n):
 
    tc = [[0 for x in range(C)] for x in range(R)]
    tc[0][0] = cost[0][0]
    for i in range(1, m+1):
        tc[i][0] = tc[i-1][0] + cost[i][0]
    for j in range(1, n+1):
        tc[0][j] = tc[0][j-1] + cost[0][j]
    for i in range(1, m+1):
        for j in range(1, n+1):
            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j]
    return tc[m][n]
 
# Driver program to test above functions
cost = [[1, 2, 3],
        [4, 8, 2],
        [1, 5, 3]]
print(minCost(cost, 2, 2))

```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/06302188-abe9-426b-b88c-0679eef384de)
```
R = int(input())
C = int(input())
import sys
def minCost(cost, m, n):
    if (n < 0 or m < 0):
        return sys.maxsize
    elif (m == 0 and n == 0):
        return cost[m][n]
    else:
        return cost[m][n] + min( minCost(cost, m-1, n-1),
                                minCost(cost, m-1, n),
                                minCost(cost, m, n-1) )

    ###########  Add your Code Here ##################
def min(x, y, z):
    if (x < y):
        return x if (x < z) else z
    else:
        return y if (y < z) else z
cost= [ [1, 2, 3],
        [4, 8, 2],
        [1, 5, 3] ]
print(minCost(cost, R-1, C-1))
```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/323263b4-f9d7-4d5d-8c8b-1e9664d520cb)
```
minSum = 1000000000
def getMinPathSum(graph, visited, necessary,
                  src, dest, currSum):
    global minSum
     
    # If destination is reached
    if (src == dest):
       
        # Set flag to true
        flag = True;
  
        # Visit all the intermediate nodes
        for i in necessary:
  
            # If any intermediate node
            # is not visited
            if (not visited[i]):
                flag = False;
                break;
     
        # If all intermediate
        # nodes are visited
        if (flag):
  
            # Update the minSum
            minSum = min(minSum, currSum);
        return;
     
    else:
  
        # Mark the current node
        # visited
        visited[src] = True;
  
        # Traverse adjacent nodes
        for node in graph[src]:
             
            if not visited[node[0]]:
             
                # Mark the neighbour visited
                visited[node[0]] = True;
 
                # Find minimum cost path
                # considering the neighbour
                # as the source
                getMinPathSum(graph, visited,
                              necessary, node[0],
                              dest, currSum + node[1]);
 
                # Mark the neighbour unvisited
                visited[node[0]] = False;
         
        # Mark the source unvisited
        visited[src] = False;
    ################# Add your Code here ################
    
if __name__=='__main__':
    graph=dict()
    graph[0] = [ [ 1, 2 ], [ 2, 3 ], [ 3, 2 ] ];
    graph[1] = [ [ 4, 4 ], [ 0, 1 ] ];
    graph[2] = [ [ 4, 5 ], [ 5, 6 ] ];
    graph[3] = [ [ 5, 7 ], [ 0, 1 ] ];
    graph[4] = [ [ 6, 4 ] ];
    graph[5] = [ [ 6, 2 ] ];
    graph[6] = [ [ 7, 11 ] ];
    n = 7;
    source = 0;
    dest = 6;
    visited=[ False for i in range(n + 1)]
    necessary = [ 2, 4 ];
    getMinPathSum(graph, visited, necessary,
                  source, dest, 0);
  
    # If no path is found
    if (minSum == 1000000000):
        print(-1)
    else:
        print(minSum)
```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/28e2880d-f99d-47be-8454-1bf89e776a7a)
```
import sys
V = 5
INF = sys.maxsize
def minimumCostSimplePath(u, destination,visited, graph):
    if (u == destination):
        return 0
  
    # Marking the current node as visited
    visited[u] = 1
  
    ans = INF
  
    # Traverse through all
    # the adjacent nodes
    for i in range(V):
        if (graph[u][i] != INF and not visited[i]):
  
            # Cost of the further path
            curr = minimumCostSimplePath(i, destination,
                                         visited, graph)
  
            # Check if we have reached the destination
            if (curr < INF):
  
                # Taking the minimum cost path
                ans = min(ans, graph[u][i] + curr)
             
    # Unmarking the current node
    # to make it available for other
    # simple paths
    visited[u] = 0
  
    # Returning the minimum cost
    return ans
########  Add your code here ##############

if __name__=="__main__":
    graph = [[INF for j in range(V)]
                  for i in range(V)]
    visited = [0 for i in range(V)]
    graph[0][1] = -1
    graph[0][3] = 1
    graph[1][2] = -2
    graph[2][0] = -3
    graph[3][2] = -1
    graph[4][3] = 2
    s = 0
    t = 2
    visited[s] = 1
    print(minimumCostSimplePath(s, t, visited, graph))
  ```
## DAY -2 Coin Change Problem
```
def count(S, n, target):
    if target == 0:
        return 1
    if target < 0 or n < 0:
        return 0
    incl = count(S, n, target - S[n])
    excl = count(S, n - 1, target)
    return incl + excl

    
    
    ###################  Add Your Code Here #################
    
    
    
    
if __name__ == '__main__':
    S = []#[1, 2, 3]
    n=int(input())
    target = int(input())
    for i in range(n):
        S.append(int(input()))
    print('The total number of ways to get the desired change is',
        count(S, len(S) - 1, target))
```
```
class Solution:
    def cherryPickup(self, grid):
        n = len(grid)
        dp = [[[None] * n for _ in range(n)] for _ in range(n)]  # Create a 3D DP array

        def f(row1, col1, col2):
            row2 = row1 + col1 - col2  # Calculate the corresponding row for the second person
            if (
                row1 == n or row2 == n or col1 == n or col2 == n  # Out of bounds
                or grid[row1][col1] == -1 or grid[row2][col2] == -1  # Blocked by thorns
            ):
                return float('-inf')  # Return a negative infinity value

            if row1 == row2 == col1 == col2 == n - 1:  # Reached the bottom-right cell
                return grid[row1][col1]  # Return the cherry count at this cell

            if dp[row1][col1][col2] is not None:  # If the result is already computed, return it
                return dp[row1][col1][col2]

            cherries = grid[row1][col1]  # Cherries collected by the first person
            if col1 != col2:  # If the two people are not at the same cell
                cherries += grid[row2][col2]  # Cherries collected by the second person

            # Move right for both people or move down for both people
            cherries += max(f(row1, col1 + 1, col2 + 1), f(row1 + 1, col1, col2), f(row1, col1 + 1, col2), f(row1 + 1, col1, col2 + 1))

            dp[row1][col1][col2] = cherries  # Store the result in the DP array
            return cherries
        #############    Add your code here  ############### 

        return max(0, f(0, 0, 0))
obj=Solution()
grid=[[0,1,-1],[1,0,-1],[1,1,1]]        
print(obj.cherryPickup(grid))
```
```
class Solution(object):
   def coinChange(self, coins, amount):
        dp = [float('inf')] * (amount + 1)  # Initialize a DP array with infinity values
        dp[0] = 0  # Base case: 0 coins needed to make an amount of 0

        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)  # Update the DP array

        return dp[amount] if dp[amount] != float('inf') else -1  # Return the f
      ####################      Add your Code Here ###########
      
      
ob1 = Solution()
n=int(input())
s=[]
amt=int(input())
for i in range(n):
    s.append(int(input()))


print(ob1.coinChange(s,amt))
```
```
def count(S, m, n):
    table = [[0 for x in range(m)] for x in range(n+1)]
    for i in range(m):
        table[0][i] = 1
    for i in range(1, n+1):
        for j in range(m):
            x = table[i - S[j]][j] if i - S[j] >= 0 else 0

            # Count of solutions excluding S[j]
            y = table[i][j - 1] if j >= 1 else 0

            # Total count
            table[i][j] = x + y

    return table[n][m - 1]
 

 
arr = []      
m = int(input())
n = int(input())
for i in range(m):
    arr.append(int(input()))
print(count(arr, m, n))
```
## DAY -3 Kadane's Algorithm
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/371eaa71-02e1-4cd1-b3b0-db0074759840)
```
def maxSubArraySum(a,size):
    max_till_now = a[0]
    max_ending = 0
    
    for i in range(0, size):
        max_ending = max_ending + a[i]
        if max_ending < 0:
            max_ending = 0
        
        
        elif (max_till_now < max_ending):
            max_till_now = max_ending
            
    return max_till_now

    
    #####################  Add your Code here #############
    
n=int(input())  
a =[] #[-2, -3, 4, -1, -2, 1, 5, -3]
for i in range(n):
    a.append(int(input()))
  
print("Maximum contiguous sum is", maxSubArraySum(a,n))


```

![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/2cd4dfcd-80e2-4f7c-bebb-7e7310806d68)
```
def maxSubArraySum(a,size):
    max_till_now = a[0]
    max_ending = 0
    
    for i in range(0, size):
        max_ending = max_ending + a[i]
        if max_ending < 0:
            max_ending = 0
        
        
        elif (max_till_now < max_ending):
            max_till_now = max_ending
            
    return max_till_now

    
    #####################  Add your Code here #############
    
n=int(input())  
a =[] #[-2, -3, 4, -1, -2, 1, 5, -3]
for i in range(n):
    a.append(int(input()))
  
print("Maximum contiguous sum is", maxSubArraySum(a,n))
```

![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/89b2208b-ed72-4887-9dc8-ce0658c457ca)
```
class Solution:
    def maxSubArray(self,A):
        max_sum = A[0]  # Initialize the maximum sum as the first element of the array
        current_sum = A[0]  # Initialize the current sum as the first element of the array

        for i in range(1, len(A)):
            current_sum = max(A[i], current_sum + A[i])  # Choose the maximum between the current element and the sum of the current element with the previous sum
            max_sum = max(max_sum, current_sum)  # Update the maximum sum if the current sum is larger

        return max_sum
        ############ Add your Code here

        
A =[]                  
n=int(input())
for i in range(n):
    A.append(int(input()))
s=Solution()
print("The sum of contiguous sublist with the largest sum is",s.maxSubArray(A))
 
```

![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/9d813a82-b87b-4b49-ad36-c5fb79f095c0)
```
class Solution:
    def maxSubArray(self,A):
        max_sum = A[0]  # Initialize the maximum sum as the first element of the array
        current_sum = A[0]  # Initialize the current sum as the first element of the array

        for i in range(1, len(A)):
            current_sum = max(A[i], current_sum + A[i])  # Choose the maximum between the current element and the sum of the current element with the previous sum
            max_sum = max(max_sum, current_sum)  # Update the maximum sum if the current sum is larger

        return max_sum

    
    #####################  Add your Code here #############
    
A =[]                  
n=int(input())
for i in range(n):
    A.append(float(input()))
s=Solution()
print("The sum of contiguous sublist with the largest sum is {:.1f}".format(s.maxSubArray(A)))
```

## DAY -4 Minimum Jump to Reach End Array
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/fe2e65c1-fa03-4ee0-9e50-012375f2491c)
```
def minJumps(arr, l, h):
    if (h == l):
        return 0
    if (arr[l] == 0):
        return float('inf')
    min = float('inf')
    for i in range(l + 1, h + 1):
        if (i < l + arr[l] + 1):
            jumps = minJumps(arr, i, h)
            if (jumps != float('inf') and
                       jumps + 1 < min):
                min = jumps + 1
 
    return min

    ###########   Add your code here ###########
    
    
arr = []
n = int(input()) 
for i in range(n):
    arr.append(int(input()))
print('Minimum number of jumps to reach','end is', minJumps(arr, 0, n-1))
 
```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/eabccf68-a289-47ca-baa5-0754576b55bf)
```
def minJumps(arr, n):
    ##########  Add your code here ##############
    jumps = [0 for i in range(n)]
 
    if (n == 0) or (arr[0] == 0):
        return float('inf')
 
    jumps[0] = 0
    for i in range(1, n):
        jumps[i] = float('inf')
        for j in range(i):
            if (i <= j + arr[j]) and (jumps[j] != float('inf')):
                jumps[i] = min(jumps[i], jumps[j] + 1)
                break
    return jumps[n-1]

arr = []
n = int(input()) #len(arr)
for i in range(n):
    arr.append(int(input()))
print('Minimum number of jumps to reach','end is', minJumps(arr,n))
 
```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/146165c6-7943-4616-81f5-01229313c47f)
```
from queue import Queue
import sys  # Import the sys module

class Pair(object):
    idx = 0
    psf = ""
    jmps = 0
    
    def __init__(self, idx, psf, jmps):
        self.idx = idx
        self.psf = psf
        self.jmps = jmps

def minJumps(arr):
    dp = [sys.maxsize] * len(arr)
    dp[-1] = 0
    
    for i in range(len(arr) - 2, -1, -1):
        steps = arr[i]
        min_jump = sys.maxsize
        
        for j in range(1, steps + 1):
            if i + j < len(arr):
                min_jump = min(min_jump, dp[i + j])
        
        if min_jump != sys.maxsize:
            dp[i] = min_jump + 1
    
    return dp

def possiblePath(arr, dp):
    queue = Queue(maxsize=0)
    p1 = Pair(0, "0", dp[0])
    queue.put(p1)
 
    while queue.qsize() > 0:
        tmp = queue.get()
 
        if tmp.jmps == 0:
            print(tmp.psf)
            continue
 
        for step in range(1, arr[tmp.idx] + 1, 1):
            if ((tmp.idx + step < len(arr)) and
               (tmp.jmps - 1 == dp[tmp.idx + step])):
               
                # Storing the neighbours
                # of current index element
                p2 = Pair(tmp.idx + step, tmp.psf +
                           " -> " + str((tmp.idx + step)),
                         tmp.jmps - 1)
                          
                queue.put(p2)

def Solution(arr):
    dp = minJumps(arr)
    possiblePath(arr, dp)
    
if __name__ == "__main__":
    arr = []
    size = int(input())
    for i in range(size):
        arr.append(int(input()))
    Solution(arr)

```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/5363d112-29cc-4d03-82bd-2a6897a06ab7)
```
def minJumps(arr, l, h):
    if (h == l):
        return 0
    if (arr[l] == 0):
        return float('inf')
    min = float('inf')
    for i in range(l + 1, h + 1):
        if (i < l + arr[l] + 1):
            jumps = minJumps(arr, i, h)
            if (jumps != float('inf') and
                       jumps + 1 < min):
                min = jumps + 1
 
    return min

    ###########   Add your code here ###########
    
    
arr = []
n = int(input()) 
for i in range(n):
    arr.append(float(input()))
print('Minimum number of jumps to reach','end is', minJumps(arr, 0, n-1))
 
```
# 22
## DAY 1 - DYNAMIC PROGRAMMING - 1
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/3b5c8dc0-e36b-4bb3-8f73-f999304d5b23)
```
def longest_common_subsequence(str1, str2):
    m = len(str1)
    n = len(str2)
    
    # Create a 2D table to store the lengths of LCSs
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the dp table using bottom-up approach
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Reconstruct the LCS using dp table
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    lcs.reverse()
    return ''.join(lcs)

# Example input
str1 = input()
str2 = input()

result = longest_common_subsequence(str1, str2)
print(result)

```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/eaf37427-b35e-45bf-a995-9ceab84d593b)
```
def longest_common_subsequence_length(X, Y):
    m = len(X)
    n = len(Y)
    
    # Create a 2D table to store the lengths of LCSs
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the dp table using bottom-up approach
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

# Example input
X = input()
Y = input()

result = longest_common_subsequence_length(X, Y)
print("Length of LCS is :", result)

```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/36d8871e-8e67-4e43-b494-a7b13814fe22)
```
# A Naive recursive Python implementation of LCS problem

def lcs(X, Y, m, n):

	if m == 0 or n == 0:
	    return 0;
	elif X[m-1] == Y[n-1]:
	    return 1 + lcs(X, Y, m-1, n-1);
	else:
	    return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n));


# Driver program to test the above function
X = input()
Y = input()
print ("Length of LCS is ", lcs(X, Y, len(X), len(Y)))

```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/46e0581f-aa5a-4097-a024-a284488e1a83)
```
def longest_common_subsequence_length(X, Y, m, n, memo):
    if m == 0 or n == 0:
        return 0
    
    if memo[m][n] != -1:
        return memo[m][n]
    
    if X[m - 1] == Y[n - 1]:
        memo[m][n] = 1 + longest_common_subsequence_length(X, Y, m - 1, n - 1, memo)
    else:
        memo[m][n] = max(longest_common_subsequence_length(X, Y, m, n - 1, memo),
                         longest_common_subsequence_length(X, Y, m - 1, n, memo))
    
    return memo[m][n]

# Wrapper function for memoization
def memoized_lcs(X, Y):
    m = len(X)
    n = len(Y)
    memo = [[-1] * (n + 1) for _ in range(m + 1)]
    return longest_common_subsequence_length(X, Y, m, n, memo)

# Example input
X = input()
Y = input()

result = memoized_lcs(X, Y)
print("Length of LCS is", result)

```
## DAY 2 - DYNAMIC PROGRAMMING - 2
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/2888b388-c814-4891-a868-105f0e893722)
```
def LCS(X, Y, m, n):
 
    maxLength = 0          
    endingIndex = m        
    lookup = [[0 for x in range(n + 1)] for y in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                lookup[i][j] = lookup[i - 1][j - 1] + 1
                if lookup[i][j] > maxLength:
                    maxLength = lookup[i][j]
                    endingIndex = i
    return X[endingIndex - maxLength: endingIndex]
if __name__ == '__main__':
    X = input()
    Y = input()
    m = len(X)
    n = len(Y)
    print('The longest common substring is', LCS(X, Y, m, n))

```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/1a8c2330-52ff-4990-8771-997b32156e12)
```
def LongComSubS(st1, st2):
  ans = 0;
  for a in range(len(st1)):
         for b in range(len(st2)):
            k = 0;
            while ((a + k) < len(st1) and (b + k) < len(st2)
        and st1[a + k] == st2[b + k]):
                k = k + 1;
            ans = max(ans, k);
  return ans;

if __name__ == '__main__':
 
    A = input()
    B = input()
    i = len(A)
    j = len(B)
    print('Length of Longest Common Substring is', LongComSubS(A, B))


```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/03a3cc68-d9a9-4588-8d73-210b68df2ac9)
```

def lcw(u, v):
    c = [[-1]*(len(v) + 1) for _ in range(len(u) + 1)]
 
    for i in range(len(u) + 1):
        c[i][len(v)] = 0
    for j in range(len(v)):
        c[len(u)][j] = 0
 
    lcw_i = lcw_j = -1
    length_lcw = 0
    for i in range(len(u) - 1, -1, -1):
        for j in range(len(v)):
            if u[i] != v[j]:
                c[i][j] = 0
            else:
                c[i][j] = 1 + c[i + 1][j + 1]
                if length_lcw < c[i][j]:
                    length_lcw = c[i][j]
                    lcw_i = i
                    lcw_j = j
 
    return length_lcw, lcw_i, lcw_j
 
    ############# Add your code here ##############
    

u = input()
v = input()
length_lcw, lcw_i, lcw_j = lcw(u, v)
print('Longest Common Subword: ', end='')
if length_lcw > 0:
    print(u[lcw_i:lcw_i + length_lcw])
    
    
    
```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/3eb20e25-6503-44a3-8792-73e46f391689)
```
def lcw(u, v):
    c = [[-1]*(len(v) + 1) for _ in range(len(u) + 1)]
    lcw_i = lcw_j = -1
    length_lcw = 0
    for i in range(len(u)):
        for j in range(len(v)):
            temp = lcw_starting_at(u, v, c, i, j)
            if length_lcw < temp:
                length_lcw = temp
                lcw_i = i
                lcw_j = j
    return length_lcw, lcw_i, lcw_j
def lcw_starting_at(u, v, c, i, j):
    if c[i][j] >= 0:
        return c[i][j]
 
    if i == len(u) or j == len(v):
        q = 0
    elif u[i] != v[j]:
        q = 0
    else:
        q = 1 + lcw_starting_at(u, v, c, i + 1, j + 1)
 
    c[i][j] = q
    return q
 
    ############ Add your code here ############

 
 
u = input()
v = input()
length_lcw, lcw_i, lcw_j = lcw(u, v)
print('Longest Common Subword: ', end='')
if length_lcw > 0:
    print(u[lcw_i:lcw_i + length_lcw])
```
## DAY 3 - DYNAMIC PROGRAMMING - 3
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/76e933ed-65f1-4af6-af03-5c67422a667b)
```
# Python 3 program of above approach

# A utility function to get max
# of two integers


def max(x, y):
	if(x > y):
		return x
	return y

# Returns the length of the longest
# palindromic subsequence in seq


def lps(seq, i, j):

	# Base Case 1: If there is
	# only 1 character
	if (i == j):
		return 1

	# Base Case 2: If there are only 2
	# characters and both are same
	if (seq[i] == seq[j] and i + 1 == j):
		return 2

	# If the first and last characters match
	if (seq[i] == seq[j]):
		return lps(seq, i + 1, j - 1) + 2

	# If the first and last characters
	# do not match
	return max(lps(seq, i, j - 1),
			lps(seq, i + 1, j))


# Driver Code
if __name__ == '__main__':
	seq = input()
	n = len(seq)
	print("The length of the LPS is",
		lps(seq, 0, n - 1))

# This code contributed by Rajput-Ji

```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/a5ed606e-6496-4273-b441-65baa4592ef2)
```
class Solution(object):
    def longestPalindrome(self, s):
        n = len(s)
        if n == 0:
            return ""

        # Create a 2D table to store palindrome information
        dp = [[False] * n for _ in range(n)]

        # Initialize single characters as palindromes
        for i in range(n):
            dp[i][i] = True

        start = 0  # Starting index of the longest palindrome found
        max_length = 1  # Maximum length of palindrome found

        # Check for palindromes of length 2
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                start = i
                max_length = 2

        # Check for palindromes of length 3 or more
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1  # Ending index of the current substring
                if dp[i + 1][j - 1] and s[i] == s[j]:
                    dp[i][j] = True
                    if length > max_length or (length == max_length and i > start):
                        start = i
                        max_length = length

        return s[start:start + max_length]

# Create an instance of the Solution class
ob1 = Solution()

# Input a string and find the longest palindromic substring using the instance
str1 = input()
print(ob1.longestPalindrome(str1))

```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/df271d85-7da0-42d2-9e40-5bc50b401313)
```
def printSubStr(str, low, high):
	
	for i in range(low, high + 1):
		print(str[i], end = "")

def longestPalindrome(str):
	############  Add your code here ###########
	n = len(str)
	
	maxLength = 1
	start = 0
	
	for i in range(n):
		for j in range(i, n):
			flag = 1
			
			for k in range(0, ((j - i) // 2) + 1):
				if (str[i + k] != str[j - k]):
					flag = 0

			if (flag != 0 and (j - i + 1) > maxLength):
				start = i
				maxLength = j - i + 1
				
	printSubStr(str, start, start + maxLength - 1)
				
#	printSubStr(str, start, start + maxLength - 1)

if __name__ == '__main__':

	str = input() 
	
	longestPalindrome(str)
```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/5a645090-b157-4e6d-8ce3-5fe9dd8d05db)
```
def expand(s, low, high):
    length = len(s)
 
    while low >= 0 and high < length and s[low] == s[high]:
        low = low - 1
        high = high + 1
 
    return s[low + 1:high]
 
 
def findLongestPalindromicSubstring(s):
 
    if not s or not len(s):
        return ''
    length = len(s)
    start = 0  # Starting index of the longest palindrome found
    end = 0    # Ending index of the longest palindrome found
 
    for i in range(length):
        # Odd length palindrome (centered at i)
        palindrome1 = expand(s, i, i)
        # Even length palindrome (centered at i and i+1)
        palindrome2 = expand(s, i, i + 1)
        
        # Find the longer palindrome between the two
        if len(palindrome1) > len(palindrome2) and len(palindrome1) > (end - start):
            start = i - len(palindrome1) // 2
            end = i + len(palindrome1) // 2
        elif len(palindrome2) > (end - start):
            start = i - len(palindrome2) // 2 + 1
            end = i + len(palindrome2) // 2
        
    return s[start:end + 1]
    ########### Add your code here ###########
 
if __name__ == '__main__':
 
    s = input()       #'mojologiccigolmojo'
 
    print(findLongestPalindromicSubstring(s))
```
## DAY 4 - DYNAMIC PROGRAMMING - 4
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/968064fb-0f07-4759-8a65-759e6da65d22)
```
def LD(s, t):
    #########  Add your code here ###########
    m = len(s)
    n = len(t)
    
    if m == 0:
        return n
    if n == 0:
        return m
    
    if s[m - 1] == t[n - 1]:
        return LD(s[:-1], t[:-1])
    
    return 1 + min(LD(s, t[:-1]),        # Insert
                   LD(s[:-1], t),        # Delete
                   LD(s[:-1], t[:-1])    # Replace
                   )
    
str1=input()
str2=input()
print('Edit Distance',LD(str1,str2))


```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/aa6fe077-0406-481c-a84a-9d572d0a66b5)
```
def edit_distance(str1, str2, a, b):
    ################ Add your code here #############
    dp = [[0] * (b + 1) for _ in range(a + 1)]
    
    # Initialize the first row and column
    for i in range(a + 1):
        dp[i][0] = i
    for j in range(b + 1):
        dp[0][j] = j
    
    for i in range(1, a + 1):
        for j in range(1, b + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],     # Insert
                                   dp[i - 1][j],     # Delete
                                   dp[i - 1][j - 1]  # Replace
                                   )
    
    return dp[a][b]
if __name__ == '__main__':
    str1 = input()
    str2 = input()
    print('No. of Operations required :',edit_distance(str1, str2, len(str1), len(str2)))
```
![image](https://github.com/SOWMIYA2003/AOA-LAB/assets/93427443/6536b1ac-3bd8-4ace-94e3-cd3776114b60)
```
def LD(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1
    res = min([LD(s[:-1], t)+1,
               LD(s, t[:-1])+1, 
               LD(s[:-1], t[:-1]) + cost])
    return res
s=input()
t=input()
print(LD(s,t))

```
# 21
# 20
## BACKTRACKING - RAT IN MAZE PROBLEM
```
N = 4
 
def printSolution( sol ):
     
    for i in sol:
        for j in i:
            print(str(j) + " ", end ="")
        print("")
 

def isSafe( maze, x, y ):
     
    if x >= 0 and x < N and y >= 0 and y < N and maze[x][y] == 1:
        return True
     
    return False
 

def solveMaze( maze ):
     
    # Creating a 4 * 4 2-D list
    sol = [ [ 0 for j in range(4) ] for i in range(4) ]
     
    if solveMazeUtil(maze, 0, 0, sol) == False:
        print("Solution doesn't exist");
        return False
     
    printSolution(sol)
    return True
     

#def solveMazeUtil(maze, x, y, sol):
#// Write your code here  

def solveMazeUtil(maze, x, y, sol):
     
    # if (x, y is goal) return True
    if x == N - 1 and y == N - 1:
        sol[x][y] = 1
        return True
         
    # Check if maze[x][y] is valid
    if isSafe(maze, x, y) == True:
        # mark x, y as part of solution path
        sol[x][y] = 1
         
        # Move forward in x direction
        if solveMazeUtil(maze, x + 1, y, sol) == True:
            return True
             
        # If moving in x direction doesn't give solution
        # then Move down in y direction
        if solveMazeUtil(maze, x, y + 1, sol) == True:
            return True
         
        # If none of the above movements work then
        # BACKTRACK: unmark x, y as part of solution path
        sol[x][y] = 0
        return False

if __name__ == "__main__":
    # Initialising the maze
    maze = [ [1, 0, 0, 0],
             [1, 1, 0, 1],
             [0, 1, 0, 0],
             [1, 1, 1, 1] ]
              
    solveMaze(maze)








```
## BACKTRACKING - NQUEEN PROBLEM 
```
global N
N = int(input())
 
def printSolution(board):
    for i in range(N):
        for j in range(N):
            print(board[i][j], end = " ")
        print()
 
def isSafe(board, row, col):
 
    # Check this row on left side
    for i in range(col):
        if board[row][i] == 1:
            return False
 
    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1),
                    range(col, -1, -1)):
        if board[i][j] == 1:
            return False
 
    # Check lower diagonal on left side
    for i, j in zip(range(row, N, 1),
                    range(col, -1, -1)):
        if board[i][j] == 1:
            return False
 
    return True
 
def solveNQUtil(board, col):
     # // Write your code here
      # Base case: If all queens are placed
    # then return true
    if col >= N:
        return True
 
    # Consider this column and try placing
    # this queen in all rows one by one
    for i in range(N):
 
        if isSafe(board, i, col):
 
            # Place this queen in board[i][col]
            board[i][col] = 1
 
            # Recur to place rest of the queens
            if solveNQUtil(board, col + 1) == True:
                return True
 
            # If placing queen in board[i][col
            # doesn't lead to a solution, then
            # queen from board[i][col]
            board[i][col] = 0
 
    # If the queen can not be placed in any row in
    # this column col then return false
    return False
 
      
def solveNQ():
    board = [ [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]
              
    if solveNQUtil(board, 0) == False:
        print ("Solution does not exist")
        return False
 
    printSolution(board)
    return True
 
# Driver Code
solveNQ()

```
## BACKTRACKING - GRAPH COLORING PROBLEM 
```
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def isSafe(self, v, colour, c):
        for i in range(self.V):
            if self.graph[v][i] and colour[i] == c:
                return False
        return True

    def graphColoringUtil(self, m, colour, v):
        if v == self.V:
            return True

        for c in range(1, m + 1):
            if self.isSafe(v, colour, c):
                colour[v] = c
                if self.graphColoringUtil(m, colour, v + 1):
                    return True
                colour[v] = 0
        return False  # Return False if no valid coloring is possible

    def graphColouring(self, m):
        colour = [0] * self.V
        if not self.graphColoringUtil(m, colour, 0):
            print("Solution does not exist")
            return
        
        return colour
            

# Driver code
g = Graph(4)
g.graph = [
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0]
]
m = 3
colors = g.graphColouring(m)

if colors:
    print("Solution exist and Following are the assigned colours:")
    for c in colors:
        print(c, end=" ")
    print()

```
## BACKTRACKING - GRAPH COLORING PROBLEM 
```
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def isSafe(self, v, colour, c):
        for i in range(self.V):
            if self.graph[v][i] and colour[i] == c:
                return False
        return True

    def graphColoringUtil(self, m, colour, v):
        if v == self.V:
            return True

        for c in range(1, m + 1):
            if self.isSafe(v, colour, c):
                colour[v] = c
                if self.graphColoringUtil(m, colour, v + 1):
                    return True
                colour[v] = 0
        return False  # Return False if no valid coloring is possible

    def graphColouring(self, m):
        colour = [0] * self.V
        if not self.graphColoringUtil(m, colour, 0):
            print("Solution does not exist")
            return
        
        return colour
            

# Driver code
g = Graph(4)
g.graph = [
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0]
]
m = 3
colors = g.graphColouring(m)

if colors:
    print("Solution exist and Following are the assigned colours:")
    for c in colors:
        print(c, end=" ")
    print()

```
# 19
