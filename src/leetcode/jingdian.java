package leetcode;

import java.util.*;

public class jingdian {
    public static void main(String[] args) {

    }

    public boolean findWhetherExistsPath(int n, int[][] graph, int start, int target) {
        List<Integer>[] adj = new ArrayList[n];
        for (int[] edge : graph) {
            int from = edge[0], to = edge[1];
            if (adj[from] == null) {
                adj[from] = new ArrayList<>();
            }
            adj[from].add(to);
        }
        LinkedList<Integer> list = new LinkedList<>();
        list.add(start);
        boolean[] visited = new boolean[n];
        visited[start] = true;
        while (!list.isEmpty()) {
            int size = list.size();
            for (int i = 0; i < size; i++) {
                int node = list.poll();
                List<Integer> nextList = adj[node];
                if (nextList == null) {
                    continue;
                }
                for (Integer next : nextList) {
                    if (next == target) return true;
                    if (visited[next]) continue;
                    visited[next] = true;
                    list.add(next);
                }
            }
        }
        return false;
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }


    int max = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        findMax(root);
        return max;
    }

    public int findMax(TreeNode root) {
        if (root == null) return 0;
        int lv = Math.max(findMax(root.left), 0);
        int rv = Math.max(findMax(root.right), 0);
        int res = root.val + lv + rv;
        max = Math.max(max, res);
        return root.val + Math.max(rv, lv);
    }

//    public TreeNode sortedArrayToBST(int[] nums) {
//        if (nums.length==0)return null;
//        TreeNode root=new TreeNode(nums[nums.length/2]);
//        root.left=sortedArrayToBST(Arrays.copyOfRange(nums,0,nums.length/2));
//        root.right=sortedArrayToBST(Arrays.copyOfRange(nums,nums.length/2+1,nums.length));
//        return root;
//    }

//    int upperlim=1;
//    int result=0;
//
//    public int totalNQueens(int n) {
//        upperlim=(1<<n)-1;
//        backtrack(0,0,0);
//        return result;
//    }
//
//    public void backtrack(long row,long ld,long rd){
//        if (upperlim!=row){
//            long pos=upperlim&~(row|ld|rd);
//            while (pos!=0){
//                long p=pos&(-pos);
//                pos-=p;
//                backtrack(row+p,(ld+p)<<1,(rd+p)>>1);
//            }
//        }else {
//            result++;
//        }
//    }

    public boolean patternMatching(String pattern, String value) {
        int countA = 0, countB = 0;
        for (char ch : pattern.toCharArray()) {
            if (ch == 'a') countA++;
            if (ch == 'b') countB++;
        }
        if (countA < countB) {
            int temp = countA;
            countA = countB;
            countB = temp;
            char[] array = pattern.toCharArray();
            for (int i = 0; i < array.length; i++) {
                array[i] = array[i] == 'a' ? 'b' : 'a';
            }
            pattern = new String(array);
        }
        if (value.length() == 0) return countB == 0;
        if (pattern.length() == 0) return false;
        for (int lenA = 0; countA * lenA <= value.length(); lenA++) {
            int rest = value.length() - countA * lenA;
            if ((countB == 0 && rest == 0) || (countB != 0 && rest % countB == 0)) {
                int lenB = (countB == 0 ? 0 : rest / countB);
                int pos = 0;
                boolean correct = true;
                String valueA = "", valueB = "";
                for (char ch : pattern.toCharArray()) {
                    if (ch == 'a') {
                        String sub = value.substring(pos, pos + lenA);
                        if (valueA.length() == 0) {
                            valueA = sub;
                        } else if (!valueA.equals(sub)) {
                            correct = false;
                            break;
                        }
                        pos += lenA;
                    } else {
                        String sub = value.substring(pos, pos + lenB);
                        if (valueB.length() == 0) {
                            valueB = sub;
                        } else if (!valueB.equals(sub)) {
                            correct = false;
                            break;
                        }
                        pos += lenB;
                    }
                }
                if (correct && !valueA.equals(valueB)) {
                    return true;
                }
            }
        }
        return false;
    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    public ListNode[] listOfDepth(TreeNode tree) {
        if (tree == null) return null;
        List<ListNode> res = new LinkedList<>();
        Queue<TreeNode> q = new LinkedList<TreeNode>() {{
            add(tree);
        }};
        while (!q.isEmpty()) {
            int size = q.size();
            ListNode head = null;
            ListNode temp = null;
            for (int i = 0; i < size; i++) {
                TreeNode node = q.poll();
                if (i == 0) {
                    head = new ListNode(node.val);
                    temp = head;
                } else {
                    ListNode cur = new ListNode(node.val);
                    temp.next = cur;
                    temp = temp.next;

                }
                if (node.left != null) q.add(node.left);
                if (node.right != null) q.add(node.right);
            }
            res.add(head);
        }
        return res.toArray(new ListNode[0]);
    }

    public boolean isBalanced(TreeNode root) {
        if (root == null) return true;
        return Math.abs(depth(root.left) - depth(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right);
    }

    public int depth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(depth(root.left), depth(root.right)) + 1;
    }

    public String addBinary(String a, String b) {
        StringBuilder sb = new StringBuilder();
        int carry = 0;
        for (int i = a.length() - 1, j = b.length() - 1; i >= 0 || j >= 0; i--, j--) {
            carry += i >= 0 ? a.charAt(i) - '0' : 0;
            carry += j >= 0 ? b.charAt(j) - '0' : 0;
            sb.append(carry % 2);
            carry /= 2;
        }
        sb.append(carry == 1 ? carry : "");
        return sb.reverse().toString();
    }

    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int ans = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length; i++) {
            int start = i + 1, end = nums.length - 1;
            while (start < end) {
                int sum = nums[i] + nums[start] + nums[end];
                if (Math.abs(sum - target) < Math.abs(ans - target)) {
                    ans = sum;
                }
                if (sum > target) {
                    end--;
                } else if (sum < target) {
                    start++;
                } else {
                    return ans;
                }
            }
        }
        return ans;
    }

    public boolean isValidBST(TreeNode root) {
        if (root == null) return true;
        TreeNode minRight = root.right, maxLeft = root.left;
        while (maxLeft != null && maxLeft.right != null) {
            maxLeft = maxLeft.right;
        }
        while (minRight != null && minRight.left != null) {
            minRight = minRight.left;
        }
        boolean res = (maxLeft == null || maxLeft.val < root.val) && (minRight == null || root.val < minRight.val);
        return res && isValidBST(root.left) && isValidBST(root.right);
    }

    public int insertBits(int N, int M, int i, int j) {
        StringBuilder sbn = new StringBuilder(Integer.toBinaryString(N)),
                sbm = new StringBuilder(Integer.toBinaryString(M));
        int remain = 32 - sbn.length();
        while (--remain >= 0) sbn.insert(0, '0');
        remain = j - i + 1 - sbm.length();
        while (--remain >= 0) sbm.insert(0, '0');
        sbn.replace(31 - j, 32 - i, sbm.toString());
        return Integer.parseInt(sbn.toString(), 2);

    }

    TreeNode res = null;

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        order(root, p, q);
        return res;
    }

    public boolean order(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return false;
        int l = order(root.left, p, q) ? 1 : 0;
        int m = (root == p || root == q) ? 1 : 0;
        int r = order(root.right, p, q) ? 1 : 0;
        if ((l + m + r) >= 2) {
            res = root;
        }
        return l + m + r >= 1;
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length()];
        Queue<Integer> q = new LinkedList<Integer>() {{
            add(0);
        }};
        while (!q.isEmpty()) {
            int start = q.poll();
            if (!dp[start]) {
                for (int end = start + 1; end <= s.length(); end++) {
                    if (set.contains(s.substring(start, end))) {
                        q.add(end);
                        if (end == s.length()) {
                            return true;
                        }
                    }
                    dp[start] = true;
                }
            }

        }
        return false;
    }

    public String printBin(double num) {
        StringBuilder sb = new StringBuilder("0.");
        for (int i = 1; i < 31 && num > 0; i++) {
            if (num >= Math.pow(0.5, i)) {
                num -= Math.pow(0.5, i);
                sb.append(1);
            } else sb.append(0);
        }
        return num == 0 ? sb.toString() : "ERROR";
    }

    public int reverseBits(int num) {
        int max = 0, pre = 0, cur = 0, bits = 32;
        while (bits-- > 0) {
            if ((num & 1) == 0) {
                cur -= pre;
                pre = cur + 1;
            }
            cur++;
            max = Math.max(max, cur);
            num >>= 1;
        }
        return max;
    }

    public ListNode removeDuplicateNodes(ListNode head) {
        Set<Integer> set = new HashSet<>();
        if (head == null) return null;
        set.add(head.val);
        ListNode pos = head;
        while (pos.next != null) {
            ListNode cur = pos.next;
            if (!set.contains(cur.val)) {
                set.add(cur.val);
                pos = pos.next;
            } else {
                pos.next = pos.next.next;
            }
        }
        pos.next = null;
        return head;
    }

    public int waysToStep(int n) {
        if (n < 4) return n == 3 ? 4 : n;
        int a = 1, b = 2, c = 4;
        for (int i = 4; i <= n; i++) {
            int temp = (a + b) % 1000000007 + c;
            a = b;
            b = c;
            c = temp % 1000000007;
        }
        return c;
    }

    int m;
    int n;
    int[][] grid;

    public List<List<Integer>> pathWithObstacles(int[][] obstacleGrid) {
        grid = obstacleGrid;
        m = grid.length;
        n = grid[0].length;
        List<List<Integer>> res = new LinkedList<>();
        dfs(0, 0, new boolean[m][n], res);
        return res;
    }

    private boolean dfs(int row, int col, boolean[][] visited, List<List<Integer>> res) {
        if (row >= m || col >= n || grid[row][col] == 1 || visited[row][col]) {
            return false;
        }
        res.add(Arrays.asList(row, col));
        if (row == m - 1 && col == n - 1) return true;
        visited[row][col] = true;
        if (dfs(row + 1, col, visited, res) || dfs(row, col + 1, visited, res)) {
            return true;
        }
        res.remove(res.size() - 1);
        return false;
    }

    int magicIndex = -1;

    public int findMagicIndex(int[] nums) {
        search(nums, 0, nums.length - 1);
        return magicIndex;
    }

    void search(int[] nums, int low, int high) {
        if (low > high) return;
        int mid = low + (high - low) / 2;
        if (nums[mid] == mid) {
            magicIndex = mid;
            search(nums, low, mid - 1);
        } else {
            search(nums, low, mid - 1);
            if (magicIndex == -1) search(nums, mid + 1, high);
        }
    }

//    public List<List<Integer>> subsets(int[] nums) {
//        List<List<Integer>> ans=new ArrayList<>();
//        for (int i=1;i<=nums.length;i++){
//            dfs(nums,new ArrayList<Integer>(),ans,i,new boolean[nums.length]);
//        }
//        ans.add(new ArrayList<Integer>());
//        return ans;
//    }
//
//    void dfs(int[] nums,List<Integer> cur,List<List<Integer>> ans,int size,boolean[] visited){
//        if (cur.size()==size){
//            ans.add(new ArrayList<>(cur));
//            return;
//        }
//        for (int i=0;i<nums.length;i++){
//            if (visited[i])break;
//            cur.add(nums[i]);
//            visited[i]=true;
//            dfs(nums,cur,ans,size,visited);
//            cur.remove(cur.size()-1);
//            visited[i]=false;
//        }
//
//    }

//    public List<List<Integer>> subsets(int[] nums) {
//        List<List<Integer>> ans=new ArrayList<>();
//        int bmp= (int) Math.pow(2,nums.length);
//        for (int i=0;i<bmp;i++){
//            List<Integer> temp=new ArrayList<>();
//            for (int j=0;j<nums.length;j++){
//                if( (i>>>j&1)==1)temp.add(nums[j]);
//            }
//            ans.add(temp);
//        }
//        return ans;
//    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        dfs(nums, ans, new ArrayList<>(), 0);
        return ans;
    }

    public void dfs(int[] nums, List<List<Integer>> ans, List<Integer> temp, int start) {
        ans.add(new ArrayList<>(temp));
        for (int i = start; i < nums.length; i++) {
            temp.add(nums[i]);
            dfs(nums, ans, temp, i + 1);
            temp.remove(temp.size() - 1);
        }
    }

    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
//            while (nums[i]>0&&nums[i]<=n&&nums[nums[i]-1]!=nums[i]){
//                int temp=nums[i];
//                nums[i]=nums[nums[i]-1];
//                nums[nums[i]-1]=temp;
//            }
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                int temp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = temp;
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }

    public int multiply(int A, int B) {
        int min = Math.min(A, B);
        int max = Math.max(A, B);
        int ans = 0;
        int i = 0;
        while (min != 0) {
            if ((min & 1) == 1) {
                ans += max << i;
            }
            i++;
            min >>= 1;
        }
        return ans;
    }

    public void hanota(List<Integer> A, List<Integer> B, List<Integer> C) {
        move(A.size(), A, B, C);
    }

    public void move(int n, List<Integer> A, List<Integer> B, List<Integer> C) {
        if (n == 1) {
            C.add(A.remove(A.size() - 1));
            return;
        }
        move(n - 1, A, C, B);
        C.add(A.remove(A.size() - 1));
        move(n - 1, B, A, C);
    }

//    public String[] permutation(String S) {
//        boolean[] visited=new boolean[S.length()];
//        ArrayList<String> ans=new ArrayList<>();
//        dfs(S,visited,new StringBuilder(),ans);
//        return ans.toArray(new String[0]);
//    }
//
//    public void dfs(String S,boolean[] visited,StringBuilder cur,ArrayList<String> ans){
//        if (cur.length()==S.length()){
//            ans.add(new String(cur));
//            return;
//        }
//        for (int i=0;i<S.length();i++){
//            if (visited[i])continue;
//            cur.append(S.charAt(i));
//            visited[i]=true;
//            dfs(S,visited,cur,ans);
//            cur.deleteCharAt(cur.length()-1);
//            visited[i]=false;
//        }
//
//    }

    public String[] permutation(String S) {
        boolean[] visited = new boolean[S.length()];
        ArrayList<String> ans = new ArrayList<>();
        char[] chars = S.toCharArray();
        Arrays.sort(chars);
        dfs(chars, visited, new StringBuilder(), ans);
        return ans.toArray(new String[0]);
    }

    public void dfs(char[] chars, boolean[] visited, StringBuilder cur, ArrayList<String> ans) {
        if (cur.length() == chars.length) {
            ans.add(new String(cur));
            return;
        }
        for (int i = 0; i < chars.length; i++) {
            if (!visited[i]) {
                if (i > 0 && chars[i] == chars[i - 1] && !visited[i - 1]) continue;
                cur.append(chars[i]);
                visited[i] = true;
                dfs(chars, visited, cur, ans);
                cur.deleteCharAt(cur.length() - 1);
                visited[i] = false;
            }
        }
    }

    public int minSubArrayLen(int s, int[] nums) {
        int ans = Integer.MAX_VALUE, lo = 0, hi = 0, sum = 0;
        while (hi < nums.length) {
            sum += nums[hi++];
            while (sum >= s) {
                ans = Math.min(ans, hi - lo);
                sum -= nums[lo++];
            }
        }
        return sum == Integer.MAX_VALUE ? 0 : ans;
    }

    public List<String> generateParenthesis(int n) {
        ArrayList<String> res = new ArrayList<>();
        dfs(0, 0, n, new StringBuilder(), res);
        return res;
    }

    private void dfs(int left, int right, int n, StringBuilder sb, ArrayList<String> res) {
        if (left == n && right == left) {
            res.add(new String(sb));
        }
        if (left < n) {
            dfs(left + 1, right, n, sb.append('('), res);
            sb.deleteCharAt(sb.length() - 1);
        }
        if (right < left) {
            dfs(left, right + 1, n, sb.append(')'), res);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int oldColor = image[sr][sc];
        dfs(image, sr, sc, newColor, oldColor);
        return image;
    }

    private void dfs(int[][] image, int sr, int sc, int newColor, int oldColor) {
        if (sr < 0 || sc < 0 || sr >= image.length || sc >= image[0].length) {
            return;
        }
        int cur = image[sr][sc];
        if (cur == oldColor && cur != newColor) {
            image[sr][sc] = newColor;
            dfs(image, sr + 1, sc, newColor, oldColor);
            dfs(image, sr - 1, sc, newColor, oldColor);
            dfs(image, sr, sc + 1, newColor, oldColor);
            dfs(image, sr, sc - 1, newColor, oldColor);

        }
    }

    public int waysToChange(int n) {
        int[] dp = new int[n + 1];
        int[] coins = {1, 5, 10, 25};
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = coin; i <= n; i++) {
                dp[i] = (dp[i] + dp[i - coin]) % 1000000007;
            }
        }
        return dp[n];
    }

    public int findKthLargest(int[] nums, int k) {
        final PriorityQueue<Integer> queue = new PriorityQueue<>();
        for (int val : nums) {
            queue.add(val);
            if (queue.size() > k)
                queue.poll();
        }
        return queue.peek();
    }


    int upperlim = 1;
    List<List<String>> ans = new ArrayList<>();

    public List<List<String>> solveNQueens(int n) {
        upperlim = (upperlim << n) - 1;
        backtrack(0, 0, 0, new ArrayList<>(), n);
        return ans;
    }

    public void backtrack(long row, long l, long r, List<String> cur, int n) {
        if (upperlim != row) {
            long position = upperlim & ~(row | l | r);
            while (position != 0) {
                long p = position & (-position);
                position -= p;
                cur.add(convert(p, n));
                backtrack(row + p, (l + p) << 1, (r + p) >> 1, cur, n);
                cur.remove(cur.size() - 1);
            }
        } else {
            ans.add(new ArrayList<>(cur));
        }
    }

    public String convert(long p, int n) {
//        String s = Long.toBinaryString(p);
//        String re = s.replace('0', '.');
//        String ans = re.replace('1', 'Q');
//        StringBuilder sb = new StringBuilder(ans);
//        sb.reverse();
//        int m=sb.length();
//        for (int i=0;i<n-m;i++){
//            sb.append('.');
//        }
//        sb.reverse();
//        return sb.toString();
        char[] res = new char[n];
        for (int i = 0; i < n; i++) {
            if (1 << i == p) {
                res[i] = 'Q';
            } else {
                res[i] = '.';
            }
        }
        return String.valueOf(res);
    }

    public int pileBox(int[][] box) {
        int len = box.length;
        Arrays.sort(box, (a, b) -> a[0] == b[0] ? a[1] == b[1] ? b[2] - a[2] : b[1] - a[1] : a[0] - b[0]);
        int[] dp = new int[len];
        dp[0] = box[0][2];
        int res = dp[0];
        for (int i = 1; i < len; i++) {
            int max = 0, depth = box[i][1], height = box[i][2];
            for (int j = 0; j < i; j++) {
                if (depth > box[j][1] && height > box[j][2]) {
                    max = Math.max(max, dp[j]);
                }
            }
            dp[i] = max + height;
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    public void merge(int[] A, int m, int[] B, int n) {
        int i = m - 1, j = n - 1, k = m + n - 1;
        while (i >= 0 && j >= 0) {
            if (A[i] >= B[j]) {
                A[k--] = A[i--];
            } else {
                A[k--] = B[j--];
            }
        }
        while (i >= 0) {
            A[k--] = A[i--];
        }
        while (j >= 0) {
            A[k--] = B[j--];
        }
    }


    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> maps = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String s = new String(chars);
            maps.computeIfAbsent(s, x -> new ArrayList<>()).add(str);
        }
        return new ArrayList<>(maps.values());
    }

    public int search(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (target == arr[i]) {
                return i;
            }
        }
        return -1;
    }

    public int findString(String[] words, String s) {
        int left = 0;
        int right = words.length - 1;
        int mid;
        while (left <= right) {
            while (words[left].length() == 0) {
                left++;
            }
            while (words[right].length() == 0) {
                right--;
            }
            mid = (left + right) / 2;
            while (mid >= 0 && words[mid].length() == 0) {
                mid--;
            }
            if (words[mid].compareTo(s) == 0) {
                return mid;
            } else if (words[mid].compareTo(s) > 0) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }

    public int findLength(int[] A, int[] B) {
        int n = A.length, m = B.length;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            int len = Math.min(m, n - i);
            int maxlen = maxLength(A, B, i, 0, len);
            ans = Math.max(ans, maxlen);
        }
        for (int i = 0; i < m; i++) {
            int len = Math.min(n, m - i);
            int maxlen = maxLength(A, B, 0, i, len);
            ans = Math.max(ans, maxlen);
        }
        return ans;
    }

    public int maxLength(int[] A, int[] B, int startA, int startB, int len) {
        int res = 0, k = 0;
        for (int i = 0; i < len; i++) {
            if (A[startA + i] == B[startB + i]) {
                k++;
            } else {
                k = 0;
            }
            res = Math.max(res, k);
        }
        return res;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return false;
        int n = matrix.length, m = matrix[0].length;
        int i = 0, j = m - 1;
        while (i >= 0 && i < n && j >= 0 && j < m) {
            if (matrix[i][j] == target) {
                return true;
            }
            if (matrix[i][j] > target) {
                j--;
                continue;
            }
            if (matrix[i][j] < target) {
                i++;
                continue;
            }
        }
        return false;
    }

    public void wiggleSort(int[] nums) {
        for (int i = 1; i < nums.length; i++) {
            if ((i % 2 == 0 && nums[i] > nums[i - 1]) || (i % 2 != 0 && nums[i] < nums[i - 1])) {
                int temp = nums[i];
                nums[i] = nums[i - 1];
                nums[i - 1] = temp;
            }
        }
    }

    public int[] swapNumbers(int[] numbers) {
        int temp = numbers[0];
        numbers[0] = numbers[1];
        numbers[1] = temp;
        return numbers;
    }

    public int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length, m = matrix[0].length;
        int l = matrix[0][0], r = matrix[n - 1][m - 1];
        while (l < r) {
            int mid = l + (r - l) / 2;
            int count = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    if (matrix[i][j] <= mid) count++;
                }
            }
            if (count < k) l = mid + 1;
            else r = mid;
        }
        return l;
    }

    public String tictactoe(String[] board) {
        int n = board.length;
        int sum_row = 0, sum_col = 0, sum_dia_left = 0, sum_dia_right = 0;
        boolean isFull = true;
        for (int i = 0; i < n; i++) {
            sum_row = 0;
            sum_col = 0;
            sum_dia_left += board[i].charAt(i);
            sum_dia_right += board[i].charAt(n - 1 - i);
            for (int j = 0; j < n; j++) {
                sum_row += board[i].charAt(j);
                sum_col += board[j].charAt(i);
                if (board[i].charAt(j) == ' ') isFull = false;
            }
            if (sum_row == ((int) 'X') * n || sum_col == ((int) 'X') * n) return "X";
            if (sum_row == ((int) 'O') * n || sum_col == ((int) 'O') * n) return "O";
        }
        if (sum_dia_left == ((int) 'X') * n || sum_dia_right == ((int) 'X') * n) return "X";
        if (sum_dia_left == ((int) 'O') * n || sum_dia_right == ((int) 'O') * n) return "O";
        if (isFull) return "Draw";
        else return "Pending";
    }

    public int trailingZeroes(int n) {
        int count = 0;
        while (n > 0) {
            n /= 5;
            count += n;
        }
        return count;
    }

    public int smallestDifference(int[] a, int[] b) {
        Arrays.parallelSort(a);
        Arrays.parallelSort(b);
        long min = Integer.MAX_VALUE;
        int i = 0, j = 0;
        while (i < a.length && j < b.length) {
            if (min == 0) return 0;
            if (a[i] < b[j]) {
                while (i < a.length && a[i] < b[j]) i++;
                if (i < a.length) min = Math.min(min, Math.abs((long) a[i] - b[j]));
                min = Math.min(min, Math.abs((long) a[i - 1] - b[j]));
            } else if (a[i] >= b[j]) {
                while (j < b.length && a[i] > b[j]) j++;
                if (j < b.length) min = Math.min(min, Math.abs((long) a[i] - b[j]));
                min = Math.min(min, Math.abs((long) a[i] - b[j - 1]));
            }
        }
        return (int) min;
    }

    /**
     * Max(a,b)=(|a-b|+a+b)/2
     *
     * @param a
     * @param b
     * @return
     */

    public int maximum(int a, int b) {
        long c = a, d = b;
        int res = (int) ((Math.abs(c - d) + c + d) / 2);
        return res;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return helper(nums, 0, nums.length - 1);
    }

    private TreeNode helper(int[] nums, int start, int end) {
        if (start > end) return null;
        int mid = (start + end) >> 1;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = helper(nums, start, mid - 1);
        node.right = helper(nums, mid + 1, end);
        return node;
    }

    public int maxAliveYear(int[] birth, int[] death) {
        int[] changes = new int[102];
        int len = birth.length, res = 1900, max_alive = 0, cur_alive = 0;
        for (int i = 0; i < len; i++) {
            changes[birth[i] - 1900]++;
            changes[death[i] - 1899]--;
        }
        for (int i = 0; i < 101; i++) {
            cur_alive += changes[i];
            if (cur_alive > max_alive) {
                max_alive = cur_alive;
                res = 1900 + i;
            }
        }
        return res;
    }

    public int[] divingBoard(int shorter, int longer, int k) {
        if (k == 0) return new int[0];
        if (shorter == longer) return new int[]{k * shorter};
        int[] res = new int[k + 1];
        for (int i = 0; i < k + 1; i++) {
            res[i] = (k - i) * shorter + i * longer;
        }
        return res;
    }

    public int[] masterMind(String solution, String guess) {
        HashMap<Character, Integer> map = new HashMap<>();
        for (char c : solution.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        int fake = 0, real = 0;
        for (char c : guess.toCharArray()) {
            if (map.containsKey(c) && map.get(c) > 0) {
                fake++;
                map.put(c, map.get(c) - 1);
            }
        }
        for (int i = 0; i < solution.length(); i++) {
            if (solution.charAt(i) == guess.charAt(i)) {
                real++;
            }
        }
        return new int[]{real, fake - real};
    }

    public int longestValidParentheses(String s) {
        int ans = 0;
        Stack<Integer> stack = new Stack<Integer>() {{
            push(-1);
        }};
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                stack.pop();
                if (stack.isEmpty()) {
                    stack.push(i);
                } else {
                    ans = Math.max(ans, i - stack.peek());
                }
            }
        }
        return ans;
    }

}

class CQueue {
    Deque<Integer> q1 = new ArrayDeque<>();
    Deque<Integer> q2 = new ArrayDeque<>();

    public CQueue() {

    }

    public void appendTail(int value) {
        q1.addLast(value);
    }

    public int deleteHead() {
        if (!q2.isEmpty()) return q2.removeLast();
        if (q1.isEmpty()) return -1;
        while (!q1.isEmpty()) {
            q2.addLast(q1.removeLast());
        }
        return q2.removeLast();
    }
}

class WordsFrequency {
    TrieNode dictionary = new TrieNode();

    public WordsFrequency(String[] book) {
        TrieNode tmp = null;
        for (int i = 0; i < book.length; i++) {
            tmp = dictionary;
            for (int j = 0; j < book[i].length(); j++) {
                if (tmp.son[book[i].charAt(j) - 'a'] == null) {
                    tmp.son[book[i].charAt(j) - 'a'] = new TrieNode();
                }
                tmp = tmp.son[book[i].charAt(j) - 'a'];
            }
            tmp.cnt++;
        }
    }

    public int get(String word) {
        TrieNode tmp = dictionary;
        for (int i = 0; i < word.length(); i++) {
            if (tmp == null) return 0;
            tmp = tmp.son[word.charAt(i) - 'a'];
        }
        return tmp == null ? 0 : tmp.cnt;
    }
}

class TrieNode {
    int cnt;
    TrieNode[] son = new TrieNode[26];

    public TrieNode() {
        this.cnt = 0;
    }
}
