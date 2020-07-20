package leetcode;


import java.util.*;

public class SSSolution {

    public static void main(String[] args) {


        LinkedHashSet<Integer> set = new LinkedHashSet<>();
        set.add(1);
        set.add(2);
        set.add(3);
        set.add(1);
        for (int n : set) {
            System.out.println(n);
        }


    }

    public int[] sortArray(int[] nums) {
        quickSort(nums, 0, nums.length - 1);
        return nums;
    }

    private void quickSort(int[] nums, int left, int right) {
        if (left > right) return;
        int temp = nums[left];
        int l = left, r = right;
        while (l < r) {
            while (l < r && nums[r] >= temp) r--;
            while (l < r && nums[l] <= temp) l++;
            if (l < r) {
                int t = nums[l];
                nums[l] = nums[r];
                nums[r] = t;
            }
        }
        nums[left] = nums[l];
        nums[l] = temp;
        quickSort(nums, left, l - 1);
        quickSort(nums, l + 1, right);
    }

    public int findDuplicate(int[] nums) {
        int l = 1, r = nums.length;
        while (l < r) {
            int mid = l + (r - l) / 2;
            int count = 0;
            for (int num : nums) {
                if (num <= mid) count++;
            }
            if (count > mid) r = mid;
            else l = mid + 1;
        }
        return l;
    }

    public boolean hasGroupsSizeX(int[] deck) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : deck) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        int res = -1;
        for (int v : map.values()) {
            if (res == -1) res = v;
            else res = gcd(res, v);
        }
        return res >= 2;
    }

    public int gcd(int x, int y) {
        return x == 0 ? y : gcd(y % x, x);
    }

    public int subarraysDivByK(int[] A, int K) {
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int sum = 0, res = 0;
        for (int num : A) {
            sum += num;
            int modules = (sum % K + K) % K;
            int same = map.getOrDefault(modules, 0);
            res += same;
            map.put(modules, same + 1);
        }
        return res;
    }

    public int minIncrementForUnique(int[] A) {
        Arrays.sort(A);
        int ans = 0, taken = 0;
        for (int i = 1; i < A.length; i++) {
            if (A[i - 1] == A[i]) {
                taken++;
                ans -= A[i];
            } else {
                int give = Math.min(taken, A[i] - A[i - 1] - 1);
                ans += give * (give + 1) / 2 + give * A[i - 1];
                taken -= give;
            }
        }
        if (A.length > 0) {
            ans += taken * (taken + 1) / 2 + taken * A[A.length - 1];
        }
        return ans;
    }

    public String decodeString(String s) {
        StringBuilder sb = new StringBuilder();
        int multi = 0;
        Stack<Integer> num = new Stack<>();
        Stack<String> str = new Stack<>();
        for (Character c : s.toCharArray()) {
            if (c == '[') {
                num.add(multi);
                str.add(sb.toString());
                multi = 0;
                sb = new StringBuilder();
            } else if (c == ']') {
                StringBuilder temp = new StringBuilder();
                int cur = num.pop();
                for (int i = 0; i < cur; i++) temp.append(sb);
                sb = new StringBuilder(str.pop() + temp);
            } else if (c >= '0' && c <= '9') multi = multi * 10 + Integer.parseInt(c + "");
            else sb.append(c);
        }
        return sb.toString();
    }

    public int orangesRotting(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        int count = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) count++;
                if (grid[i][j] == 2) queue.add(new int[]{i, j});
            }
        }
        int time = 0;
        while (count > 0 && !queue.isEmpty()) {
            time++;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] orange = queue.poll();
                int r = orange[0];
                int c = orange[1];
                if (r - 1 >= 0 && grid[r - 1][c] == 1) {
                    grid[r - 1][c] = 2;
                    queue.add(new int[]{r - 1, c});
                    count--;
                }
                if (r + 1 < m && grid[r + 1][c] == 1) {
                    grid[r + 1][c] = 2;
                    queue.add(new int[]{r + 1, c});
                    count--;
                }
                if (c - 1 >= 0 && grid[r][c - 1] == 1) {
                    grid[r][c - 1] = 2;
                    queue.add(new int[]{r, c - 1});
                    count--;
                }
                if (c + 1 < n && grid[r][c + 1] == 1) {
                    grid[r][c + 1] = 2;
                    queue.add(new int[]{r, c + 1});
                    count--;
                }
            }
        }
        if (count > 0) return -1;
        else return time;
    }

    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        if (nums.length <= 2) return Math.max(nums[0], nums[1]);
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(nums[i] + dp[i - 2], dp[i - 1]);
        }
        return dp[nums.length - 1];
    }

    public int numRookCaptures(char[][] board) {
        int[] dx = {-1, 1, 0, 0};
        int[] dy = {0, 0, -1, 1};
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (board[i][j] == 'R') {
                    int res = 0;
                    for (int k = 0; k < 4; k++) {
                        int x = i, y = j;
                        while (true) {
                            x += dx[k];
                            y += dy[k];
                            if (x < 0 || x >= 8 || y < 0 || y >= 8 || board[x][y] == 'B') break;
                            if (board[x][y] == 'p') {
                                res++;
                                break;
                            }
                        }
                    }
                    return res;
                }
            }
        }
        return 0;
    }

    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int[] left = new int[n];
        int[] right = new int[n];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && heights[stack.peek()] >= heights[i]) {
                stack.pop();
            }
            left[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }
        stack = new Stack<>();
        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && heights[stack.peek()] >= heights[i]) {
                stack.pop();
            }
            right[i] = stack.isEmpty() ? n : stack.peek();
            stack.push(i);
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans = Math.max(ans, heights[i] * (right[i] - left[i] - 1));
        }
        return ans;
    }

    public boolean canThreePartsEqualSum(int[] A) {
        int sum = 0;
        for (int a : A) sum += a;
        if (sum % 3 != 0) return false;
        int leftSum = A[0], rightSum = A[A.length - 1];
        int i = 0, j = A.length - 1;
        while (i + 1 < j) {
            if (leftSum == sum / 3 && rightSum == sum / 3) return true;
            if (leftSum != sum / 3) leftSum += A[++i];
            if (rightSum != sum / 3) rightSum += A[--j];
        }
        return false;
    }

    public int lastRemaining(int n, int m) {
        int res = 0;
        for (int i = 2; i <= n; i++) {
            res = (res + m) % i;
        }
        return res;
    }

    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) set.add(num);
        int max = 0;
        for (int num : set) {
            if (!set.contains(num - 1)) {
                int curNum = num;
                int cur = 1;
                while (set.contains(curNum + 1)) {
                    curNum++;
                    cur += 1;
                }
                max = Math.max(cur, max);
            }
        }
        return max;
    }

    public int myAtoi(String str) {
        char[] chars = str.toCharArray();
        int n = chars.length;
        int idx = 0;
        while (idx < n && chars[idx] == ' ') idx++;
        if (idx == n) return 0;
        boolean negative = false;
        if (chars[idx] == '-') {
            negative = true;
            idx++;
        } else if (chars[idx] == '+') {
            idx++;
        } else if (!Character.isDigit(chars[idx])) {
            return 0;
        }
        int ans = 0;
        while (idx < n && Character.isDigit(chars[idx])) {
            int digit = chars[idx] - '0';
            if (ans > (Integer.MAX_VALUE - digit) / 10) {
                return negative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
            }
            ans = ans * 10 + digit;
            idx++;
        }
        return negative ? -ans : ans;
    }

    public int maxArea(int[] height) {
        int res = 0;
        int i = 0, j = height.length - 1;
        while (i < j) {
            int area = (j - i) * Math.min(height[i], height[j]);
            res = Math.max(res, area);
            if (height[i] < height[j]) {
                i++;
            } else {
                j--;
            }
        }
        return res;
    }

    private static final int INF = 1 << 20;
    private Map<String, Integer> wordId;
    private ArrayList<String> idWord;
    private ArrayList<Integer>[] edges;

    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        wordId = new HashMap<>();
        idWord = new ArrayList<>();
        int id = 0;
        // 将wordList所有单词加入wordId中 相同的只保留一个 // 并为每一个单词分配一个id
        for (String word : wordList) {
            if (!wordId.containsKey(word)) {
                wordId.put(word, id++);
                idWord.add(word);
            }
        }
        // 若endWord不在wordList中 则无解
        if (!wordId.containsKey(endWord)) {
            return new ArrayList<>();
        }
        // 把beginWord也加入wordId中
        if (!wordId.containsKey(beginWord)) {
            wordId.put(beginWord, id++);
            idWord.add(beginWord);
        }

        // 初始化存边用的数组
        edges = new ArrayList[idWord.size()];
        for (int i = 0; i < idWord.size(); i++) {
            edges[i] = new ArrayList<>();
        }
        // 添加边
        for (int i = 0; i < idWord.size(); i++) {
            for (int j = i + 1; j < idWord.size(); j++) {
                // 若两者可以通过转换得到 则在它们间建一条无向边
                if (transformCheck(idWord.get(i), idWord.get(j))) {
                    edges[i].add(j);
                    edges[j].add(i);
                }
            }
        }

        int dest = wordId.get(endWord); // 目的ID
        List<List<String>> res = new ArrayList<>(); // 存答案
        int[] cost = new int[id]; // 到每个点的代价
        for (int i = 0; i < id; i++) {
            cost[i] = INF; // 每个点的代价初始化为无穷大
        }

        // 将起点加入队列 并将其cost设为0
        Queue<ArrayList<Integer>> q = new LinkedList<>();
        ArrayList<Integer> tmpBegin = new ArrayList<>();
        tmpBegin.add(wordId.get(beginWord));
        q.add(tmpBegin);
        cost[wordId.get(beginWord)] = 0;

        // 开始广度优先搜索
        while (!q.isEmpty()) {
            ArrayList<Integer> now = q.poll();
            int last = now.get(now.size() - 1); // 最近访问的点
            if (last == dest) { // 若该点为终点则将其存入答案res中
                ArrayList<String> tmp = new ArrayList<>();
                for (int index : now) {
                    tmp.add(idWord.get(index)); // 转换为对应的word
                }
                res.add(tmp);
            } else { // 该点不为终点 继续搜索
                for (int i = 0; i < edges[last].size(); i++) {
                    int to = edges[last].get(i);
                    // 此处<=目的在于把代价相同的不同路径全部保留下来
                    if (cost[last] + 1 <= cost[to]) {
                        cost[to] = cost[last] + 1;
                        // 把to加入路径中
                        ArrayList<Integer> tmp = new ArrayList<>(now);
                        tmp.add(to);
                        q.add(tmp); // 把这个路径加入队列
                    }
                }
            }
        }
        return res;
    }

    // 两个字符串是否可以通过改变一个字母后相等
    boolean transformCheck(String str1, String str2) {
        int differences = 0;
        for (int i = 0; i < str1.length() && differences < 2; i++) {
            if (str1.charAt(i) != str2.charAt(i)) {
                ++differences;
            }
        }
        return differences == 1;
    }

//    public boolean equationsPossible(String[] equations) {
//        int length=equations.length;
//        int[] parent=new int[26];
//        for (int i=0;i<26;i++){
//            parent[i]=-1;
//        }
//        for (String str:equations){
//            if (str.charAt(1)=='='){
//                int index1=str.charAt(0)-'a';
//                int index2=str.charAt(3)-'a';
//                union(parent,index1,index2);
//            }
//        }
//        for (String str:equations){
//            if (str.charAt(1)=='!'){
//                int index1=str.charAt(0)-'a';
//                int index2=str.charAt(3)-'a';
//                if (find(parent,index1)==find(parent,index2)){
//                    return false;
//                }
//            }
//        }
//        return true;
//    }
//
//    public void union(int[] parent,int index1,int index2){
////        parent[find(parent,index1)]=find(parent,index2);
//        int root1=find(parent,index1);
//        int root2=find(parent,index2);
//        if (root1==root2)return;
//        parent[root1]=root2;
//    }
//
//
//    public int find(int[] parent,int index){
////        while (parent[index]!=index){
////            parent[index]=parent[parent[index]];
////            index=parent[index];
////        }
////        return index;
//        int temp=index;
//        while (parent[temp]!=-1){
//            temp=parent[temp];
//        }
//        return temp;
//    }

    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        if (n == 0) return res;
        StringBuilder sb = new StringBuilder();
        dfs(sb, n, n, res);
        return res;
    }

    private void dfs(StringBuilder cur, int left, int right, List<String> res) {
        if (left == 0 && right == 0) {
            res.add(cur.toString());
            return;
        }
        if (left > right) return;
        if (left > 0) {
            cur.append("(");
            dfs(cur, left - 1, right, res);
            cur.deleteCharAt(cur.length() - 1);
        }
        if (right > 0) {
            cur.append(")");
            dfs(cur, left, right - 1, res);
            cur.deleteCharAt(cur.length() - 1);
        }
    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    public ListNode mergeKLists(ListNode[] lists) {
        Queue<ListNode> pq = new PriorityQueue<>((v1, v2) -> v1.val - v2.val);
        for (ListNode node : lists) {
            if (node != null) {
                pq.offer(node);
            }
        }
        ListNode res = new ListNode(-1);
        ListNode cur = res;
        while (!pq.isEmpty()) {
            ListNode temp = pq.poll();
            cur.next = temp;
            cur = cur.next;
            if (temp.next != null) {
                pq.offer(temp.next);
            }
        }
        return res.next;
//        ListNode dummyHead = new ListNode(0);
//        ListNode tail = dummyHead;
//        while (!pq.isEmpty()) {
//            ListNode minNode = pq.poll();
//            tail.next = minNode;
//            tail = minNode;
//            if (minNode.next != null) {
//                pq.offer(minNode.next);
//            }
//        }
//        return dummyHead.next;
    }

    public int translateNum(int num) {
        String s = String.valueOf(num);
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= s.length(); i++) {
            String temp = s.substring(i - 2, i);
            if (Integer.valueOf(temp) >= 10 && Integer.valueOf(temp) <= 25) {
                dp[i] = dp[i - 1] + dp[i - 2];
            } else {
                dp[i] = dp[i - 1];
            }
        }
        return dp[s.length()];
    }

    public int search(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        int mid = l + (r - l) / 2;
        while (l <= r) {
            if (nums[mid] == target) return mid;
            if (nums[mid] >= nums[l]) {
                if (target >= nums[l] && target <= nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (target >= nums[mid] && target <= nums[r]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
            mid = l + (r - l) / 2;
        }
        return -1;
    }

    public int trap(int[] height) {
        int ans = 0;
        int[] left = new int[height.length];
        int[] right = new int[height.length];
        for (int i = 1; i < height.length; i++) {
            left[i] = Math.max(left[i - 1], height[i - 1]);
        }
        for (int i = height.length - 2; i >= 1; i--) {
            right[i] = Math.max(right[i + 1], height[i + 1]);
        }
        for (int i = 1; i < height.length - 1; i++) {
            int temp = Math.min(left[i], right[i]);
            if (temp > height[i]) {
                ans += temp - height[i];
            }
        }
        return ans;
    }

    public boolean isPalindrome(int x) {
        return new StringBuilder(String.valueOf(x)).reverse().toString().equals(String.valueOf(x));
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtrack(res, nums, new ArrayList<>(), new boolean[nums.length]);
        return res;
    }

    public void backtrack(List<List<Integer>> res, int[] nums, List<Integer> cur, boolean[] b) {
        if (cur.size() == nums.length) {
            res.add(new ArrayList<>(cur));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (b[i]) {
                continue;
            }
            cur.add(nums[i]);
            b[i] = true;
            backtrack(res, nums, cur, b);
            cur.remove(cur.size() - 1);
            b[i] = false;
        }
    }

    public boolean canJump(int[] nums) {
        int max = 0;
        for (int i = 0; i < nums.length; i++) {
            if (max >= i && (i + nums[i]) > max) {
                max = i + nums[i];
            }
        }
        return max >= nums.length - 1;
    }

    public int[] dailyTemperatures(int[] T) {
        int length = T.length;
        int[] ans = new int[length];
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < length; i++) {
            int temp = T[i];
            while (!stack.isEmpty() && temp > T[stack.peekFirst()]) {
                int index = stack.pop();
                ans[index] = i - index;
            }
            stack.push(i);
        }
        return ans;
    }

    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (o1, o2) -> o1[0] - o2[0]);
        int[][] ans = new int[intervals.length][2];
        int idx = -1;
        for (int[] interval : intervals) {
            if (idx == -1 || interval[0] > ans[idx][1]) {
                ans[++idx] = interval;
            } else {
                ans[idx][1] = Math.max(ans[idx][1], interval[1]);
            }
        }
        return Arrays.copyOf(ans, idx + 1);
    }

    public int minDistance(String word1, String word2) {
        int n1 = word1.length();
        int n2 = word2.length();
        int[][] dp = new int[n1 + 1][n2 + 1];
        for (int j = 1; j <= n2; j++) dp[0][j] = dp[0][j - 1] + 1;
        for (int i = 1; i <= n1; i++) dp[i][0] = dp[i - 1][0] + 1;
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) dp[i][j] = dp[i - 1][j - 1];
                else dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i][j - 1]), dp[i - 1][j]) + 1;
            }
        }
        return dp[n1][n2];
    }

//    public int singleNumber(int[] nums) {
//        int[] counts=new int[32];
//        for (int num:nums){
//            for (int j=0;j<32;j++){
//                counts[j]+=num&1;
//                num>>>=1;
//            }
//        }
//        int res=0,m=3;
//        for (int i=0;i<32;i++){
//            res<<=1;
//            res|=counts[31-i]%m;
//        }
//        return res;
//    }

    public int[] singleNumber(int[] nums) {
        int temp = 0;
        for (int num : nums) temp ^= num;
        temp &= (-temp);
        int[] res = new int[2];
        for (int num : nums) {
            if ((num & temp) == 0) res[0] ^= num;
            else res[1] ^= num;
        }
        return res;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < n - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int j = i + 1, k = n - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum < 0) {
                    while (j < k && nums[j] == nums[++j]) ;
                } else if (sum > 0) {
                    while (j < k && nums[k] == nums[--k]) ;
                } else {
                    res.add(new ArrayList<>(Arrays.asList(nums[i], nums[j], nums[k])));
                    while (j < k && nums[j] == nums[++j]) ;
                    while (j < k && nums[k] == nums[--k]) ;
                }
            }
        }
        return res;
    }

    public String reverseWords(String s) {
        s.trim();
        StringBuilder sb = new StringBuilder();
        int j = s.length() - 1, i = j;
        while (i >= 0) {
            while (i >= 0 && s.charAt(i) != ' ') i--;
            sb.append(s.substring(i + 1, j + 1) + " ");
            while (i >= 0 && s.charAt(i) == ' ') i--;
            j = i;
        }
        return sb.toString().trim();
    }

    public int climbStairs(int n) {
        if (n == 1) return 1;
        int first = 1;
        int second = 2;
        for (int i = 3; i <= n; i++) {
            int third = first + second;
            first = second;
            second = third;
        }
        return second;
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> integers = new ArrayList<>();
        dfs(root, 0, integers);
        return integers;
    }

    public void dfs(TreeNode root, int depth, List<Integer> res) {
        if (root == null) return;
        if (depth == res.size()) res.add(root.val);
        depth++;
        dfs(root.right, depth, res);
        dfs(root.left, depth, res);
    }

    public int numIslands(char[][] grid) {
        if (grid.length == 0) return 0;
        int x = grid.length;
        int y = grid[0].length;
        int[] nums = new int[x * y];
        Arrays.fill(nums, -1);
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                if (grid[i][j] == '1') {
                    grid[i][j] = '0';
                    if (i < (x - 1) && grid[i + 1][j] == '1') {
                        union(nums, i * y + j, (i + 1) * y + j);
                    }
                    if (j < (y - 1) && grid[i][j + 1] == '1') {
                        union(nums, i * y + j, i * y + j + 1);
                    }
                } else nums[i * y + j] = -2;
            }
        }
        int count = 0;
        for (int num : nums) {
            if (num == -1) count++;
        }
        return count;
    }

    public int find(int[] parents, int i) {
        if (parents[i] == -1) return i;
        return find(parents, parents[i]);
    }

    public void union(int[] parents, int x, int y) {
        int xset = find(parents, x);
        int yset = find(parents, y);
        if (xset != yset) {
            parents[xset] = yset;
        }
    }

    public boolean isHappy(int n) {
        int slow = n;
        int fast = getNext(n);
        while (fast != 1 && slow != fast) {
            slow = getNext(slow);
            fast = getNext(getNext(fast));
        }
        return fast == 1;
    }

    public int getNext(int n) {
        int sum = 0;
        while (n > 0) {
            int d = n % 10;
            n = n / 10;
            sum += d * d;
        }
        return sum;
    }

    public int findBestValue(int[] arr, int target) {
        int max = 0, sum = 0;
        for (int a : arr) {
            sum += a;
            max = Math.max(max, a);
        }
        if (sum <= target) return max;
        int ans = target / arr.length;
        sum = getSum(arr, ans);
        while (sum < target) {
            int temp = getSum(arr, ans + 1);
            if (temp >= target) return (target - sum) <= (temp - target) ? ans : ans + 1;
            sum = temp;
            ans++;
        }
        return ans;
    }

    public int getSum(int[] arr, int value) {
        int sum = 0;
        for (int i : arr) sum += i < value ? i : value;
        return sum;
    }

    public void gameOfLife(int[][] board) {
        int[] neighbors = {-1, 0, 1};
        int rows = board.length;
        int cols = board[0].length;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int liveNeighbors = 0;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        if (!(neighbors[i] == 0 && neighbors[j] == 0)) {
                            int r = row + neighbors[i];
                            int c = col + neighbors[j];
                            if ((r >= 0 && r < rows) && (c >= 0 && c < cols) && (Math.abs(board[r][c]) == 1)) {
                                liveNeighbors++;
                            }
                        }
                    }
                }
                if (board[row][col] == 1 && (liveNeighbors < 2 || liveNeighbors > 3)) {
                    board[row][col] = -1;
                }
                if (board[row][col] == 0 && liveNeighbors == 3) {
                    board[row][col] = 2;
                }
            }
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i][j] > 0) board[i][j] = 1;
                else board[i][j] = 0;
            }
        }
    }


//    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
//        Stack<Integer> stack1=new Stack<>();
//        Stack<Integer> stack2=new Stack<>();
//        while (l1!=null){
//            stack1.push(l1.val);
//            l1=l1.next;
//        }
//        while (l2!=null){
//            stack2.push(l2.val);
//            l2=l2.next;
//        }
//        int carry=0;
//        ListNode head=null;
//        while (!stack1.isEmpty()||!stack2.isEmpty()||carry>0){
//            int sum=carry;
//            sum+=stack1.isEmpty()?0:stack1.pop();
//            sum+=stack2.isEmpty()?0:stack2.pop();
//            ListNode node=new ListNode(sum%10);
//            node.next=head;
//            head=node;
//            carry=sum/10;
//        }
//        return head;
//    }

    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode temp = cur.next;
            cur.next = prev;
            prev = cur;
            cur = temp;
        }
        return prev;
    }

    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) return "";
        String ans = strs[0];
        for (int i = 1; i < strs.length; i++) {
            int j = 0;
            for (; j < ans.length() && j < strs[i].length(); j++) {
                if (ans.charAt(j) != strs[i].charAt(j)) break;
            }
            ans = ans.substring(0, j);
            if (ans.equals("")) return ans;
        }
        return ans;
    }

    public int[][] updateMatrix(int[][] matrix) {
        Queue<int[]> queue = new LinkedList<>();
        int m = matrix.length, n = matrix[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    queue.offer(new int[]{i, j});
                } else {
                    matrix[i][j] = -1;
                }
            }
        }
        int[] dx = {-1, 1, 0, 0};
        int[] dy = {0, 0, -1, 1};
        while (!queue.isEmpty()) {
            int[] point = queue.poll();
            int x = point[0], y = point[1];
            for (int i = 0; i < 4; i++) {
                int newX = x + dx[i];
                int newY = y + dy[i];
                if (newX >= 0 && newX < m && newY >= 0 && newY < n && matrix[newX][newY] == -1) {
                    matrix[newX][newY] = matrix[x][y] + 1;
                    queue.offer(new int[]{newX, newY});
                }
            }
        }
        return matrix;
    }

    public int[] singleNumbers(int[] nums) {
        int[] res = new int[2];
        int temp = 0;
        for (int num : nums) temp ^= num;
        temp = temp & (-temp);
        for (int num : nums) {
            if ((temp & num) == 0) res[0] ^= num;
            else res[1] ^= num;
        }
        return res;
    }

//    public void rotate(int[][] matrix) {
//        int n=matrix.length;
//        for (int i=0;i<n-1;i++){
//            for(int j=i+1;j<n;j++){
//                int temp=matrix[i][j];
//                matrix[i][j]=matrix[j][i];
//                matrix[j][i]=temp;
//            }
//        }
//        int mid=n>>1;
//        for(int i=0;i<n;i++){
//            for (int j = 0; j < mid; j++) {
//                int temp=matrix[i][j];
//                matrix[i][j]=matrix[i][n-1-j];
//                matrix[i][n-1-j]=temp;
//            }
//        }
//    }

    public boolean isUnique(String astr) {
        int mark = 0;
        for (char c : astr.toCharArray()) {
            int move = c - 'a';
            if ((mark & (1 << move)) != 0) return false;
            else mark |= (1 << move);
        }
        return true;
    }

    public boolean CheckPermutation(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        int[] data = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            data[s1.charAt(i) - 'a']++;
            data[s2.charAt(i) - 'a']--;
        }
        for (int d : data) {
            if (d != 0) return false;
        }
        return true;
    }

    public String replaceSpaces(String S, int length) {
        int i = 0, j = 0;
        char[] ans = new char[length * 3];
        while (i < length) {
            char c = S.charAt(i);
            if (c == ' ') {
                ans[j++] = '%';
                ans[j++] = '2';
                ans[j++] = '0';
            } else {
                ans[j++] = c;
            }
            i++;
        }
        return new String(ans, 0, j);
    }

    public boolean canPermutePalindrome(String s) {
        int[] data = new int[128];
        for (char c : s.toCharArray()) {
            data[c]++;
        }
        int flag = 0;
        for (int d : data) {
            if (d % 2 == 1) flag++;
        }
        return flag <= 1;
    }

    public boolean oneEditAway(String first, String second) {
        int n = first.length(), m = second.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= m; i++) dp[0][i] = i;
        for (int i = 1; i <= n; i++) dp[i][0] = i;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (first.charAt(i - 1) == second.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i - 1][j - 1], dp[i][j - 1])) + 1;
                }
            }
        }
        return dp[n][m] <= 1;
    }

    public int maxScoreSightseeingPair(int[] A) {
        int ans = 0, max = A[0] + 0;
        for (int i = 1; i < A.length; i++) {
            ans = Math.max(ans, max + A[i] - i);
            max = Math.max(max, A[i] + i);
        }
        return ans;
    }

    public String compressString(String S) {
        StringBuilder sb = new StringBuilder();
        int n = S.length();
        int i = 0;
        while (i < n) {
            int j = i;
            while (j < n && S.charAt(i) == S.charAt(j)) j++;
            sb.append(i);
            sb.append(j - i);
            i = j;
        }
        String res = sb.toString();
        return res.length() >= S.length() ? S : res;
    }

    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][n - 1 - j];
                matrix[i][n - 1 - j] = temp;
            }
        }
    }

    public int smallestRepunitDivByK(int K) {
        if (K % 2 == 0 || K % 5 == 0) return -1;
        int temp = 1, ans = 1;
        while (temp % K != 0) {
            temp %= K;
            temp = temp * 10 + 1;
            ans++;
        }
        return ans;
    }

    public void setZeroes(int[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;
        boolean[] col = new boolean[m];
        boolean[] row = new boolean[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (matrix[i][j] == 0) {
                    row[i] = true;
                    col[j] = true;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            if (row[i]) {
                for (int j = 0; j < m; j++) {
                    matrix[i][j] = 0;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            if (col[i]) {
                for (int j = 0; j < n; j++) {
                    matrix[j][i] = 0;
                }
            }
        }
    }

    public TreeNode recoverFromPreorder(String S) {
        Deque<TreeNode> path = new LinkedList<>();
        int pos = 0;
        while (pos < S.length()) {
            int level = 0;
            while (S.charAt(pos) == '-') {
                ++level;
                ++pos;
            }
            int value = 0;
            while (pos < S.length() && Character.isDigit(S.charAt(pos))) {
                value = value * 10 + (S.charAt(pos) - '0');
                ++pos;
            }
            TreeNode node = new TreeNode(value);
            if (level == path.size()) {
                if (!path.isEmpty()) {
                    path.peek().left = node;
                }
            } else {
                while (level != path.size()) {
                    path.pop();
                }
                path.peekFirst().right = node;
            }
            path.push(node);
        }
        while (path.size() > 1) {
            path.pop();
        }
        return path.peek();
    }

    public boolean isFlipedString(String s1, String s2) {
        return s1.length() == s2.length() && (s1 + s1).contains(s2);
    }

    public ListNode removeDuplicateNodes(ListNode head) {
        LinkedHashSet<Integer> set = new LinkedHashSet<>();
        while (head != null) {
            set.add(head.val);
            head = head.next;
        }
        ListNode res = new ListNode(-1);
        ListNode cur = res;
        for (int n : set) {
            ListNode temp = new ListNode(n);
            cur.next = temp;
            cur = cur.next;
        }
        return res.next;
    }

    public int kthToLast(ListNode head, int k) {
        ListNode temp = head;
        for (int i = 0; i < k; i++) {
            temp = temp.next;
        }
        while (temp != null) {
            temp = temp.next;
            head = head.next;
        }
        return head.val;
    }

    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    public ListNode partition(ListNode head, int x) {
        ListNode l = new ListNode(-1);
        ListNode r = new ListNode(-1);
        ListNode l1 = l;
        ListNode r1 = r;
        while (head != null) {
            if (head.val < x) {
                l.next = head;
                l = l.next;
            } else {
                r.next = head;
                r = r.next;
            }
            head = head.next;
        }
        r.next = null;
        l.next = r1.next;
        return l1.next;
    }

    public boolean isPalindrome(String s) {
        int n = s.length();
        int left = 0, right = n - 1;
        while (left < right) {
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            }
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            }
            if (left < right) {
                if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
                    return false;
                }
                left++;
                right--;
            }
        }
        return true;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode ans = new ListNode(-1);
        ListNode cur = ans;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int i1 = l1 == null ? 0 : l1.val;
            int i2 = l2 == null ? 0 : l2.val;
            int sum = i1 + i2 + carry;
            l1 = l1 == null ? null : l1.next;
            l2 = l2 == null ? null : l2.next;
            ListNode temp = new ListNode(sum % 10);
            cur.next = temp;
            cur = cur.next;
            carry = sum / 10;
        }
        if (carry != 0) {
            cur.next = new ListNode(carry);
        }
        return ans.next;
    }

    public boolean isPalindrome(ListNode head) {
        if (head == null) return true;
        ListNode fast = head, slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode cur = slow;
        ListNode prev = null;
        while (cur != null) {
            ListNode temp = cur.next;
            cur.next = prev;
            prev = cur;
            cur = temp;
        }
        while (head != null && prev != null) {
            if (head.val != prev.val) return false;
            head = head.next;
            prev = prev.next;
        }
        return true;
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode a = headA, b = headB;
        while (a != b) {
            a = a == null ? headB : a.next;
            b = b == null ? headA : b.next;
        }
        return a;
    }

    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) return null;
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) break;
        }
        if (fast == null || fast.next == null) return null;
        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    public boolean isMatch(String A, String B) {
        int n = A.length(), m = B.length();
        boolean[][] dp = new boolean[n + 1][m + 1];
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                if (j == 0) {
                    dp[i][j] = i == 0;
                } else {
                    if (B.charAt(j - 1) != '*') {
                        if (i > 0 && (A.charAt(i - 1) == B.charAt(j - 1) || B.charAt(j - 1) == '.')) {
                            dp[i][j] = dp[i - 1][j - 1];
                        }
                    } else {
                        if (j >= 2) {
                            dp[i][j] |= dp[i][j - 2];
                        }
                        if (i >= 1 && j >= 2 && (A.charAt(i - 1) == B.charAt(j - 2) || B.charAt(j - 2) == '.')) {
                            dp[i][j] |= dp[i - 1][j];
                        }
                    }
                }
            }
        }
        return dp[n][m];
    }
}
