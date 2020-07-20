package leetcode;


import java.util.*;

class Solution {

    public static void main(String[] args) {

    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    int maxVal = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        findMax(root);
        return maxVal;
    }

    public int findMax(TreeNode root) {
        if (root == null) return 0;
        int lv = Math.max(findMax(root.left), 0);
        int rv = Math.max(findMax(root.right), 0);
        int res = root.val + lv + rv;
        maxVal = Math.max(res, maxVal);
        return root.val + Math.max(lv, rv);
    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }


    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode res = new ListNode(-1);
        ListNode temp = res;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int n1 = l1 == null ? 0 : l1.val;
            int n2 = l2 == null ? 0 : l2.val;
            int sum = (n1 + n2 + carry) % 10;
            carry = sum / 10;
            l1 = l1.next;
            l2 = l2.next;
            temp = temp.next;
        }
        if (carry != 0) {
            temp.next = new ListNode(carry);

        }
        return res.next;
    }

//    public int[] levelOrder(TreeNode root) {
//        if (root==null)return new int[0];
//        Queue<TreeNode> q=new LinkedList<TreeNode>();
//        q.add(root);
//        ArrayList<Integer> arr=new ArrayList<>();
//        while (!q.isEmpty()){
//            TreeNode poll = q.poll();
//            arr.add(poll.val);
//            if (poll.left!=null)q.add(poll.left);
//            if (poll.right!=null)q.add(poll.right);
//        }
//        return arr.stream().mapToInt(Integer::intValue).toArray();
//    }

    public int maxSubArray(int[] nums) {
        int max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i - 1] > 0) nums[i] += nums[i - 1];
            max = Math.max(max, nums[i]);
        }
        return max;
    }

//    public ListNode reverseList(ListNode head) {
//        if (head==null||head.next==null)return head;
//        ListNode pre=null;
//        ListNode cur=head;
//        while(cur!=null){
//            ListNode next=cur.next;
//            cur.next=pre;
//            pre=cur;
//            cur=next;
//        }
//        return pre;
//    }

    public static int subarraySum(int[] nums, int k) {
        int sum = 0, res = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int n : nums) {
            sum += n;
            if (map.containsKey(sum - k)) {
                res += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return res;
    }

    public int singleNumber(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int n : nums) {
            map.put(n, map.getOrDefault(n, 0) + 1);
        }
        for (Map.Entry<Integer, Integer> i : map.entrySet()) {
            if (i.getValue() == 1)
                return i.getKey();
        }
        return -1;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return false;
        int m = matrix.length, n = matrix[0].length;
        int row = 0, col = n - 1;
        while (row < m && col >= 0) {
            if (target == matrix[row][col]) return true;
            else if (target > matrix[row][col]) row++;
            else col--;
        }
        return false;
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int[] copy = new int[m];
        System.arraycopy(nums1, 0, copy, 0, m);
        int m1 = 0, n1 = 0, i = 0;
        while (m1 < m && n1 < n) {
            nums1[i++] = copy[m1] < nums2[n1] ? copy[m1++] : nums2[n1++];
        }
        if (m1 < m) System.arraycopy(copy, m1, nums1, i, m - m1);
        if (n1 < n) System.arraycopy(nums2, n1, nums1, i, n - n1);
    }

    public boolean isPalindrome(String s) {
        String str = s.toLowerCase();
        StringBuilder sb = new StringBuilder();
        for (char c : str.toCharArray()) {
            if (Character.isLetterOrDigit(c)) sb.append(c);
        }
        return sb.toString().equals(sb.reverse().toString());
    }


    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        back(s, new ArrayList(), res);
        return res;
    }

    public void back(String s, ArrayList temp, List<List<String>> res) {
        if (s == null || s.length() == 0) {
            res.add(new ArrayList<>(temp));
            return;
        }
        for (int i = 1; i <= s.length(); i++) {
            if (isP(s.substring(0, i))) {
                temp.add(s.substring(0, i));
                back(s.substring(i), temp, res);
                temp.remove(temp.size() - 1);
            }
        }
    }

    public boolean isP(String s) {
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            sb.append(c);
        }
        return sb.toString().equals(sb.reverse().toString());
    }

//    public static boolean wordBreak(String s, List<String> wordDict) {
//        HashSet<String> set=new HashSet<>(wordDict);
//        Queue<Integer> queue=new ArrayDeque<>();
//        boolean[] b=new boolean[s.length()];
//        queue.add(0);
//        while (!queue.isEmpty()){
//            int start=queue.remove();
//            if (!b[start]){
//                for (int end=start+1;end<=s.length();end++){
//                    if (set.contains(s.substring(start,end))){
//                        queue.add(end);
//                        if (end==s.length())
//                            return true;
//                    }
//                }
//            b[start]=true;
//            }
//        }
//        return false;
//    }

    public List<String> wordBreak(String s, List<String> wordDict) {
        HashMap<Integer, List<String>> map = new HashMap<>();
        return wB(s, 0, new HashSet<>(wordDict), map);
    }

    public List<String> wB(String s, int start, HashSet<String> set, HashMap<Integer, List<String>> map) {
        if (map.containsKey(start)) return map.get(start);
        List<String> res = new ArrayList<>();
        if (start == s.length()) res.add("");
        for (int end = start + 1; end <= s.length(); end++) {
            if (set.contains(s.substring(start, end))) {
                List<String> list = wB(s, end, set, map);
                for (String st : list) {
                    res.add(s.substring(start, end) + (st.equals("") ? "" : " ") + st);
                }
            }
        }
        map.put(start, res);
        return res;
    }

    public boolean isAnagram(String s, String t) {
        int[] a = new int[26];

        for (char c : s.toCharArray()) {
            a[c - 'a']++;
        }
        for (char c : t.toCharArray()) {
            a[c - 'a']--;
        }
        for (int i : a) {
            if (i != 0)
                return false;
        }
        return true;
    }

    public int firstUniqChar(String s) {
        LinkedHashMap<Character, Integer> l = new LinkedHashMap<>();
        for (char c : s.toCharArray()) {
            l.put(c, l.getOrDefault(c, 0) + 1);
        }
        for (Map.Entry<Character, Integer> set : l.entrySet()) {
            if (set.getValue() == 1) {
                return s.indexOf(set.getKey());
            }
        }
        return -1;
    }

    public void reverseString(char[] s) {
        int i = 0, j = s.length - 1;
        while (i < j) {
            char temp = s[i];
            s[i] = s[j];
            s[j] = temp;
            i++;
            j--;
        }
    }

    public String reverseStr(String s, int k) {
        char[] chars = s.toCharArray();
        for (int i = 0; i < chars.length; i += 2 * k) {
            int l = i, r = Math.min(i + k - 1, chars.length - 1);
            while (l < r) {
                char temp = chars[l];
                chars[l++] = chars[r];
                chars[r++] = temp;
            }
        }
        return chars.toString();
    }

    public ListNode deleteNode(ListNode head, int val) {
        if (head == null) return null;
        ListNode prev = head;
        if (head.val == val) return head.next;
        ListNode temp = head.next;
        while (temp != null) {
            if (temp.val == val) {
                prev.next = temp.next;
            }
            prev = prev.next;
            temp = temp.next;
        }
        return head;
    }

    public boolean isMatch(String A, String B) {
        int n = A.length();
        int m = B.length();
        boolean[][] f = new boolean[n + 1][m + 1];
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                if (j == 0) {
                    f[i][j] = i == 0;
                } else {
                    if (B.charAt(j - 1) != '*') {
                        if (i > 0 && (A.charAt(i - 1) == B.charAt(j - 1) || B.charAt(j - 1) == '.')) {
                            f[i][j] = f[i - 1][j - 1];
                        }
                    } else {
                        if (j >= 2) {
                            f[i][j] |= f[i][j - 2];
                        }
                        //看
                        if (i >= 1 && j >= 2 && (A.charAt(i - 1) == B.charAt(j - 2) || B.charAt(j - 2) == '.')) {
                            f[i][j] |= f[i - 1][j];
                        }

//                            if (j>=2){
//                                f[i][j]|=f[i][j-2];
//                            }
//                            if (i>=1&&j>=2&&A.charAt(i-1)==B.charAt(j-2)||B.charAt(j-2)=='.'){
//                                f[i][j]|=f[i-1][j];
//                            }
                    }
                }
            }
        }
        return f[n][m];
    }

    public boolean isNumber(String s) {
        if (s == null || s.length() == 0) return false;
        boolean num = false;
        boolean dot = false;
        boolean e = false;
        char[] str = s.trim().toCharArray();
        for (int i = 0; i < str.length; i++) {
            if (str[i] >= '0' && str[i] <= '9') {
                num = true;
            } else if (str[i] == '.') {
                if (dot || e) {
                    return false;
                }
                dot = true;
            } else if (str[i] == 'e' || str[i] == 'E') {
                if (e || !num) {
                    return false;
                }
                e = true;
                num = false;
            } else if (str[i] == '+' || str[i] == '-') {
                if (i != 0 && str[i - 1] != 'e' && str[i - 1] != 'E') {
                    return false;
                }
            } else {
                return false;
            }
        }
        return num;
    }

    public int[] exchange(int[] nums) {
        int i = 0, j = nums.length - 1, temp;
        while (i < j) {
            while (i < j && (nums[i] % 2 == 1)) i++;
            while (i < j && (nums[j] % 2 == 0)) j--;
            temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }
        return nums;
    }

    public ListNode getKthFromEnd(ListNode head, int k) {

        ListNode fast = head, slow = head;
        for (int i = 0; i < k; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;

    }

    public ListNode reverseList(ListNode head) {
        ListNode cur = head;
        ListNode temp = null;
        ListNode pre = null;
        while (cur != null) {
            temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }
        return pre;
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode res = new ListNode(0);
        ListNode temp = res;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                temp.next = l1;
                l1 = l1.next;
                temp = temp.next;
            } else {
                temp.next = l2;
                l2 = l2.next;
                temp = temp.next;
            }
        }
        if (l1 != null) {
            temp.next = l1;
        }
        if (l2 != null) {
            temp.next = l2;
        }
        return res.next;
    }

    public boolean isSubStructure(TreeNode A, TreeNode B) {
        return (A != null && B != null) && (recur(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B));
    }

    public static boolean recur(TreeNode A, TreeNode B) {
        if (B == null) return true;
        if (A == null || A.val != B.val) return false;
        return recur(A.left, B.left) && recur(A.right, B.right);
    }

    public TreeNode mirrorTree(TreeNode root) {
        mirror(root);
        return root;
    }

    public static void mirror(TreeNode root) {
        if (root == null) return;
        if (root.left != null || root.right != null) {
            TreeNode temp = root.left;
            root.left = root.right;
            root.right = temp;
        }
        mirror(root.left);
        mirror(root.right);
    }

    public boolean isSymmetric(TreeNode root) {
        return root == null ? true : judge(root.left, root.right);
    }

    public static boolean judge(TreeNode L, TreeNode R) {
        if (L == null && R == null) return true;
        if (L == null || R == null || L.val != R.val) return false;
        return judge(L.left, R.right) && judge(L.right, R.left);
    }

    public int[] spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return new int[]{};
        ArrayList<Integer> res = new ArrayList<>();
        int l = 0, r = matrix[0].length - 1, u = 0, d = matrix.length - 1;
        while (true) {
            for (int i = l; i <= r; i++) res.add(matrix[u][i]);
            if (++u > d) break;
            for (int i = u; i <= d; i++) res.add(matrix[i][r]);
            if (--r < l) break;
            for (int i = r; i >= l; i--) res.add(matrix[d][i]);
            if (--d < u) break;
            for (int i = d; i >= u; i--) res.add(matrix[i][l]);
            if (++l > r) break;
        }
        return res.stream().mapToInt(Integer::intValue).toArray();
    }

    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> s = new Stack<>();
        int i = 0;
        for (int num : pushed) {
            s.push(num);
            while (!s.isEmpty() && s.peek() == popped[i]) {
                s.pop();
                i++;
            }
        }
        return s.isEmpty();
    }

    public int[] levelOrder1(TreeNode root) {
        if (root == null) return new int[]{};
        Queue<TreeNode> q = new ArrayDeque<TreeNode>() {{
            add(root);
        }};
        ArrayList<Integer> res = new ArrayList<>();
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            res.add(node.val);
            if (node.left != null) q.add(node.left);
            if (node.right != null) q.add(node.right);
        }
        return res.stream().mapToInt(Integer::intValue).toArray();
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> q = new ArrayDeque<TreeNode>();
        if (root != null) q.add(root);
        while (!q.isEmpty()) {
            List<Integer> temp = new ArrayList<>();
            for (int i = q.size(); i > 0; i--) {
                TreeNode node = q.poll();
                temp.add(node.val);
                if (node.left != null) q.add(node.left);
                if (node.right != null) q.add(node.right);
            }
            if (res.size() % 2 == 1) {
                Collections.reverse(temp);
            }
            res.add(temp);
        }
        return res;
    }

    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        recur(root, sum);
        return res;
    }

    public void recur(TreeNode root, int sum) {
        if (root == null) return;
        path.add(root.val);
        sum -= root.val;
        if (sum == 0 && root.left == null && root.right == null)
            res.add(new ArrayList<>(path));
        recur(root.left, sum);
        recur(root.right, sum);
        path.remove(path.size() - 1);
    }

    class Node {
        public int val;
        public Node left;
        public Node right;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right) {
            val = _val;
            left = _left;
            right = _right;
        }
    }

//    public Node copyRandomList(Node head) {
//        if(head==null)return null;
//        for (Node node=head,copy=null;node!=null;node=node.next.next){
//            copy=new Node(node.val);
//            copy.next=node.next;
//            node.next=copy;
//        }
//        for (Node node=head;node!=null;node=node.next.next){
//            if (node.random!=null){
//                node.next.random=node.random.next;
//            }
//        }
//        Node newHead=head.next;
//        for (Node node=head,temp=null;node!=null&&node.next!=null;){
//            temp=node.next;
//            node.next=temp.next;
//            node=temp;
//        }
//        return newHead;
//    }

    Node pre, head;

    public Node treeToDoublyList(Node root) {
        if (root == null) return null;
        dfs(root);
        pre.right = head;
        head.left = pre;
        return head;
    }

    void dfs(Node cur) {
        if (cur == null) return;
        dfs(cur.left);
        if (pre != null) pre.right = cur;
        else head = cur;
        cur.left = pre;
        pre = cur;
        dfs(cur.right);
    }


    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) return "[]";
        StringBuilder res = new StringBuilder("[");
        Queue<TreeNode> q = new LinkedList<TreeNode>() {{
            add(root);
        }};
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            if (node != null) {
                res.append(node.val + ",");
                q.add(node.left);
                q.add(node.right);
            } else res.append("null,");
        }
        res.deleteCharAt(res.length() - 1);
        res.append("]");
        return res.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.equals("[]")) return null;
        String[] strings = data.substring(1, data.length() - 1).split(",");
        TreeNode root = new TreeNode(Integer.valueOf(strings[0]));
        Queue<TreeNode> q = new LinkedList<TreeNode>() {{
            add(root);
        }};
        int i = 1;
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            if (!strings[i].equals("null")) {
                node.left = new TreeNode(Integer.valueOf(strings[i]));
                q.add(node.left);
            }
            i++;
            if (!strings[i].equals("null")) {
                node.right = new TreeNode(Integer.valueOf(strings[i]));
                q.add(node.right);
            }
            i++;
        }
        return root;
    }


    public String[] permutation(String s) {
        List<String> res = new LinkedList<>();
        boolean[] visited = new boolean[s.length()];
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        dfss(chars, res, visited, new StringBuilder());
        return res.toArray(new String[res.size()]);

    }

    void dfss(char[] chars, List<String> res,
              boolean[] visited, StringBuilder cur) {
        if (chars.length == cur.length()) {
            res.add(cur.toString());
            return;
        }
        for (int i = 0; i < chars.length; i++) {
            if (visited[i]) continue;
            if (i > 0 && chars[i - 1] == chars[i] && !visited[i - 1]) continue;
            visited[i] = true;
            cur.append(chars[i]);
            dfss(chars, res, visited, cur);
            visited[i] = false;
            cur.deleteCharAt(cur.length() - 1);
        }
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new LinkedList<>();
        Arrays.sort(nums);
        for (int i = 0; i <= nums.length; i++)
            dfsss(res, nums, new LinkedList<>(), i, 0);
        return res;
    }

    void dfsss(List<List<Integer>> res,
               int[] nums, LinkedList<Integer> cur, int len, int start) {
        if (cur.size() == len) {
            res.add(new ArrayList<>(cur));
        }
        for (int i = start; i < nums.length; i++) {
            if (i != start && nums[i] == nums[i - 1]) continue;
            cur.add(nums[i]);
            dfsss(res, nums, cur, len, i + 1);
            cur.removeLast();
        }
    }

    public int majorityElement(int[] nums) {
        int k = 0;
        int temp = nums[0];
        for (int num : nums) {
            if (k == 0) temp = num;
            k += num == temp ? 1 : -1;
        }
        return temp;
    }

//    public int[] getLeastNumbers(int[] arr, int k) {
//        if (k==0)return new int[]{};
//        PriorityQueue<Integer> p = new PriorityQueue<>(k, Comparator.reverseOrder());
//        for (int num:arr){
//            if (p.isEmpty()||p.size()<k||num<p.peek()){
//                p.add(num);
//            }
//            if (p.size()>k){
//                p.poll();
//            }
//        }
//        return p.stream().mapToInt(Integer::intValue).toArray();
//    }

    public int[] getLeastNumbers(int[] arr, int k) {
        if (k == 0 || arr.length == 0) return new int[]{};
        return quickSearch(arr, 0, arr.length - 1, k - 1);
    }

    private int[] quickSearch(int[] nums, int lo, int hi, int k) {
        int j = partition(nums, lo, hi);
        if (j == k) return Arrays.copyOf(nums, j + 1);
        return j > k ? quickSearch(nums, lo, j - 1, k) : quickSearch(nums, j + 1, hi, k);
    }

    private int partition(int[] nums, int lo, int hi) {
        int base = nums[lo];
        int i = lo, j = hi;
        while (i < j) {
            while (nums[j] >= base && i < j) j--;
            while (nums[i] <= base && i < j) i++;
            if (i < j) {
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
            }
        }
        nums[lo] = nums[i];
        nums[i] = base;
        return i;
    }

}

class MinStack {
    Stack<Integer> A, B;

    /**
     * initialize your data structure here.
     */
    public MinStack() {
        A = new Stack<>();
        B = new Stack<>();
    }

    public void push(int x) {
        A.add(x);
        if (B.isEmpty() || B.peek() >= x) B.add(x);
    }

    public void pop() {
        if (A.pop().equals(B.peek())) B.pop();
    }

    public int top() {
        return A.peek();
    }

    public int min() {
        return B.peek();
    }
}