package leetcode;

import java.util.*;
import java.util.stream.Collectors;

public class jingdian2 {

    public static void main(String[] args) {

    }

    public int[] subSort(int[] array) {
        if (array == null || array.length == 0) return new int[]{-1, -1};
        int last = -1, first = -1;
        int max = Integer.MIN_VALUE, min = Integer.MAX_VALUE;
        int len = array.length;
        for (int i = 0; i < len; i++) {
            if (array[i] < max) {
                last = i;
            } else {
                max = Math.max(max, array[i]);
            }
            if (array[len - i - 1] > min) {
                first = len - i - 1;
            } else {
                min = Math.min(min, array[len - i - 1]);
            }
        }
        return new int[]{first, last};
    }

    public int maxSubArray(int[] nums) {
        int max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i - 1] > 0) {
                nums[i] += nums[i - 1];
            }
            max = Math.max(max, nums[i]);
        }
        return max;
    }

    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= n; ++i) {
            if (p.charAt(i - 1) == '*') {
                dp[0][i] = true;
            } else {
                break;
            }
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
                } else if (p.charAt(j - 1) == '?' || s.charAt(i - 1) == p.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }

    public int[] pondSizes(int[][] land) {
        ArrayList<Integer> res = new ArrayList<>();
        int n = land.length, m = land[0].length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (land[i][j] > 0) continue;
                int size = 0;
                size = dfs(land, i, j);
                if (size > 0) res.add(size);
            }
        }
        int[] ans = res.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(ans);
        return ans;
    }

    public int dfs(int[][] land, int row, int col) {
        int size = 0;
        if (row < 0 || row >= land.length || col < 0 || col >= land[0].length || land[row][col] != 0) {
            return size;
        }
        size++;
        land[row][col] = -1;
        size += dfs(land, row + 1, col);
        size += dfs(land, row - 1, col);
        size += dfs(land, row + 1, col + 1);
        size += dfs(land, row + 1, col - 1);
        size += dfs(land, row, col + 1);
        size += dfs(land, row, col - 1);
        size += dfs(land, row - 1, col + 1);
        size += dfs(land, row - 1, col - 1);
        return size;
    }

    public List<String> getValidT9Words(String num, String[] words) {
        HashMap<Character, Character> map = new HashMap<>();
        map.put('a', '2');
        map.put('b', '2');
        map.put('c', '2');
        map.put('d', '3');
        map.put('e', '3');
        map.put('f', '3');
        map.put('g', '4');
        map.put('h', '4');
        map.put('i', '4');
        map.put('j', '5');
        map.put('k', '5');
        map.put('l', '5');
        map.put('m', '6');
        map.put('n', '6');
        map.put('o', '6');
        map.put('p', '7');
        map.put('q', '7');
        map.put('r', '7');
        map.put('s', '7');
        map.put('t', '8');
        map.put('u', '8');
        map.put('v', '8');
        map.put('w', '9');
        map.put('x', '9');
        map.put('y', '9');
        map.put('z', '9');
        List<String> ans = new ArrayList<>();
        for (String word : words) {
            boolean match = true;
            for (int i = 0; i < word.length(); i++) {
                char n = num.charAt(i);
                char c = word.charAt(i);
                if (map.get(c) != n) {
                    match = false;
                    break;
                }
            }
            if (match) {
                ans.add(word);
            }
        }
        return ans;
    }

    public int[] findSwapValues(int[] array1, int[] array2) {
        int diff = Arrays.stream(array1).sum() - Arrays.stream(array2).sum();
        if ((diff & 1) == 0) {
            diff >>= 1;
            Arrays.sort(array1);
            Set<Integer> collect = Arrays.stream(array2).boxed().collect(Collectors.toSet());
            for (int num : collect) {
                int idx = Arrays.binarySearch(array1, num + diff);
                if (idx >= 0) {
                    return new int[]{array1[idx], num};
                }
            }
        }
        return new int[0];
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int n = obstacleGrid.length, m = obstacleGrid[0].length;
        if (m == 0 || n == 0) return 0;
        int[][] dp = new int[n][m];
        for (int i = 0; i < n; i++) {
            if (obstacleGrid[i][0] == 1) break;
            else dp[i][0] = 1;
        }
        for (int i = 0; i < m; i++) {
            if (obstacleGrid[0][i] == 1) break;
            else dp[0][i] = 1;
        }
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < m; j++) {
                if (obstacleGrid[i][j] == 1) dp[i][j] = 0;
                else dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[n - 1][m - 1];
    }

    public List<List<Integer>> pairSums(int[] nums, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        int start = 0, end = nums.length - 1;
        while (start < end) {
            int sum = nums[start] + nums[end];
            if (sum < target) {
                start++;
            } else if (sum > target) {
                end--;
            } else {
                ans.add(Arrays.asList(nums[start], nums[end]));
                start++;
                end--;
            }
        }
        return ans;
    }

    public int calculate(String s) {
        if (s == null || s.length() == 0) return 0;
        s = s.replace(" ", "");
        Stack<Integer> stack = new Stack<>();
        char[] chars = s.toCharArray();
        int i = 0;
        while (i < chars.length) {
            char temp = chars[i];
            if (temp == '*' || temp == '/' || temp == '+' || temp == '-') {
                i++;
            }
            int num = 0;
            while (i < chars.length && Character.isDigit(chars[i])) {
                num = num * 10 + chars[i] - '0';
                i++;
            }
            switch (temp) {
                case '-':
                    num = -num;
                    break;
                case '*':
                    num = stack.pop() * num;
                    break;
                case '/':
                    num = stack.pop() / num;
                    break;
            }
            stack.push(num);
        }
        int res = 0;
        while (!stack.isEmpty()) {
            res += stack.pop();
        }
        return res;
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) return false;
        if (root.left == null && root.right == null && root.val == sum) return true;
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }

    public int add(int a, int b) {
        while (b != 0) {
            int sum = (a ^ b);
            int carry = (a & b) << 1;
            a = sum;
            b = carry;
        }
        return a;
    }

    public int[] divingBoard(int shorter, int longer, int k) {
        if (k == 0) return new int[0];
        if (shorter == longer) return new int[]{shorter * k};
        int[] ans = new int[k + 1];
        for (int i = 0; i < k + 1; i++) {
            ans[i] = longer * i + shorter * (k - i);
        }
        return ans;
    }

    public int missingNumber(int[] nums) {
        int res = 0;
        for (int n : nums) {
            res ^= n;
        }
        for (int i = 0; i < nums.length + 1; i++) {
            res ^= i;
        }
        return res;
    }

    public String[] findLongestSubarray(String[] array) {
        int len = array.length;
        int[] memo = new int[(len << 1) + 1];
        Arrays.fill(memo, -2);
        memo[len] = -1;
        int res = 0, count = 0, begin = 0, end = 0;
        for (int i = 0; i < len; i++) {
            char c = array[i].toCharArray()[0];
            count += Character.isDigit(c) ? -1 : 1;
            if (memo[count + len] <= -2) memo[count + len] = i;
            else if (i - memo[count + len] > res) {
                begin = memo[count + len] + 1;
                end = i + 1;
                res = i - memo[count + len];
            }
        }
        return Arrays.copyOfRange(array, begin, end);
    }

//    public int respace(String[] dictionary, String sentence) {
//        HashSet<String> set=new HashSet<>();
//        for (String s:dictionary)set.add(s);
//        int len=sentence.length();
//        int[] dp=new int[len+1];
//        for (int i=1;i<=len;i++){
//            dp[i]=dp[i-1]+1;
//            for (int j=i-1;j>=0;j--){
//                String str=sentence.substring(j,i);
//                if (set.contains(str)){
//                    dp[i]=Math.min(dp[i],dp[j]);
//                }else {
//                    dp[i]=Math.min(dp[i],dp[j]+i-j);
//                }
//            }
//        }
//        return dp[len];
//    }

    public String[] trulyMostPopular(String[] names, String[] synonyms) {
        Map<String, Integer> map = new HashMap<>();
        Map<String, String> unionMap = new HashMap<>();
        for (String name : names) {
            int idx1 = name.indexOf('(');
            int idx2 = name.indexOf(')');
            int frequency = Integer.valueOf(name.substring(idx1 + 1, idx2));
            map.put(name.substring(0, idx1), frequency);
        }
        for (String pair : synonyms) {
            int idx = pair.indexOf(',');
            String name1 = pair.substring(1, idx);
            String name2 = pair.substring(idx + 1, pair.length() - 1);
            while (unionMap.containsKey(name1)) {
                name1 = unionMap.get(name1);
            }
            while (unionMap.containsKey(name2)) {
                name2 = unionMap.get(name2);
            }
            if (!name1.equals(name2)) {
                int frequency = map.getOrDefault(name1, 0) + map.getOrDefault(name2, 0);
                String father = name1.compareTo(name2) < 0 ? name1 : name2;
                String son = name1.compareTo(name2) < 0 ? name2 : name1;
                unionMap.put(son, father);
                map.remove(son);
                map.put(father, frequency);
            }
        }
//        String[] res=new String[map.size()];
        ArrayList<String> ans = new ArrayList<>();
        int index = 0;
        for (String name : map.keySet()) {
            StringBuilder sb = new StringBuilder(name);
            sb.append('(');
            sb.append(map.get(name));
            sb.append(')');
//            res[index++]=sb.toString();
            ans.add(sb.toString());
        }
//        return res;
        return ans.toArray(new String[0]);
    }

    public int bestSeqAtIndex(int[] height, int[] weight) {
        int len = height.length;
        int[][] people = new int[len][2];
        for (int i = 0; i < len; i++) {
            people[i] = new int[]{height[i], weight[i]};
        }
        Arrays.sort(people, (a, b) -> a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]);
        int[] dp = new int[len + 1];
        int res = 0;
        for (int[] pair : people) {
            int i = Arrays.binarySearch(dp, 0, res, pair[1]);
            if (i < 0) {
                i = -(i + 1);
            }
            dp[i] = pair[1];
            if (i == res) res++;
        }
        return res;
    }

    public int maxProfit(int[] prices) {
        // 0:hold shares
        // 1:not hold shares and in freezing period
        // 2:not hold shares and not in freezing period
        if (prices.length == 0) return 0;
        int len = prices.length;
        int[][] dp = new int[len][3];
        dp[0][0] = -prices[0];
        for (int i = 1; i < len; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][2] - prices[i]);
            dp[i][1] = dp[i - 1][0] + prices[i];
            dp[i][2] = Math.max(dp[i - 1][1], dp[i - 1][2]);
        }
        return Math.max(dp[len - 1][1], dp[len - 1][2]);
    }

    public int getKthMagicNumber(int k) {
        int p3 = 0, p5 = 0, p7 = 0;
        int[] dp = new int[k];
        dp[0] = 1;
        for (int i = 1; i < k; i++) {
            dp[i] = Math.min(dp[p3] * 3, Math.min(dp[p5] * 5, dp[p7] * 7));

            if (dp[i] == dp[p3] * 3) p3++;
            if (dp[i] == dp[p5] * 5) p5++;
            if (dp[i] == dp[p7] * 7) p7++;
        }
        return dp[k - 1];
    }

    public int majorityElement(int[] nums) {
        int temp = nums[0];
        int count = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == temp) count++;
            else count--;
            if (count == 0) {
                temp = nums[i];
                count = 1;
            }
        }
        int t = nums.length / 2 + 1;
        count = 0;
        for (int num : nums) {
            if (num == temp) count++;
            if (count == t) return temp;
        }
        return -1;
    }

    public int findClosest(String[] words, String word1, String word2) {
        int min = Integer.MAX_VALUE, i1 = -1, i2 = -1;
        for (int i = 0; i < words.length; i++) {
            if (words[i].equals(word1)) i1 = i;
            if (words[i].equals(word2)) i2 = i;
            if (i1 == -1 || i2 == -1) continue;
            int tmp = i1 > i2 ? i1 - i2 : i2 - i1;
            min = Math.min(tmp, min);
        }
        return min;
    }

    public List<Integer> countSmaller(int[] nums) {
        int n = nums.length;
        List<Integer> list = new ArrayList<>();
        int[] res = new int[n];
        for (int i = n - 1; i >= 0; i--) {
            if (i == n - 1) {
                list.add(nums[i]);
                res[i] = 0;
            } else {
                int index = binaryS(list, nums[i]);
                list.add(index, nums[i]);
                res[i] = index;
            }
        }
        return Arrays.stream(res).boxed().collect(Collectors.toList());
    }

    public int binaryS(List<Integer> nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums.get(mid) >= target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    public TreeNode convertBiNode(TreeNode root) {
        TreeNode head = new TreeNode(-1);
        TreeNode prev = head;
        TreeNode node = root;
        Stack<TreeNode> stack = new Stack<>();
        while (node != null || !stack.isEmpty()) {
            if (node != null) {
                stack.push(node);
                node = node.left;
            } else {
                node = stack.pop();
                node.left = null;
                prev.right = node;
                prev = node;
                node = node.right;
            }
        }
        return head.right;
    }

    public int respace(String[] dictionary, String sentence) {
        HashSet<String> set = new HashSet<>();
        for (String s : dictionary) set.add(s);
        int[] dp = new int[sentence.length() + 1];
        for (int i = 1; i <= sentence.length(); i++) {
            dp[i] = dp[i - 1] + 1;
            for (int j = i - 1; j >= 0; j--) {
                String temp = sentence.substring(j, i);
                if (set.contains(temp)) {
                    dp[i] = Math.min(dp[i], dp[j]);
                } else {
                    dp[i] = Math.min(dp[i], dp[j] + i - j);
                }
            }
        }
        return dp[sentence.length()];
    }

    public int[] smallestK(int[] arr, int k) {
        PriorityQueue<Integer> stack = new PriorityQueue<>(Comparator.reverseOrder());
        for (int a : arr) {
            stack.add(a);
            if (stack.size() > k) {
                stack.poll();
            }
        }
        return stack.stream().mapToInt(Integer::intValue).toArray();
    }

    public int[] intersect(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map=new HashMap<>();
        for (int n:nums1){
            int count=map.getOrDefault(n,0)+1;
            map.put(n,count);
        }
        int[] ans=new int[Math.min(nums1.length,nums2.length)];
        int index=0;
        for (int n:nums2){
            int count=map.getOrDefault(n,0);
            if (count>0){
                ans[index++]=n;
                count--;
                map.put(n,count);
            }
        }
        return Arrays.copyOfRange(ans,0,index);
    }

    public String longestWord(String[] words) {
        Arrays.sort(words,(o1, o2)->{
            if (o1.length()==o2.length()){
                return o1.compareTo(o2);
            }else {
                return o2.length()-o1.length();
            }
        });
        Set<String> set=new HashSet<>(Arrays.asList(words));
        for (String s:words){
            set.remove(s);
            if(find(set,s)){
                return s;
            }
        }
        return "";
    }

    public boolean find(Set<String> set, String word){
        if (word.length()==0)return true;
        for (int i=0;i<word.length();i++){
            if (set.contains(word.substring(0,i+1))&&find(set,word.substring(i+1))){
                return true;
            }
        }
        return false;
    }

    public int massage(int[] nums) {
        if (nums.length==0)return 0;
        if (nums.length==1)return nums[0];
        if(nums.length==2)return Math.max(nums[0],nums[1]);
        int[] dp=new int[nums.length];
        dp[0]=nums[0];
        dp[1]= Math.max(nums[0],nums[1]);
        for (int i=2;i<nums.length;i++){
            dp[i]= Math.max(dp[i-1],dp[i-2]+nums[i]);
        }
        return dp[nums.length-1];
    }

    public int[][] multiSearch(String big, String[] smalls) {
        int len=smalls.length;
        int[][] res=new int[len][];
        for (int i=0;i<len;i++){
            helper(i,smalls[i],res,big);
        }
        return res;
    }

    private void helper(int position, String str, int[][] res, String big){
        if (str.equals("")){
            res[position]=new int[0];
            return;
        }
        LinkedList<Integer> list=new LinkedList<>();
        int index=0;
        while ((index=big.indexOf(str,index)+1)!=0){
            list.add(index-1);
        }
        res[position]=list.stream().mapToInt(Integer::intValue).toArray();
    }

    public int[] shortestSeq(int[] big, int[] small) {
        int left=0,right=0,start=0,min= Integer.MAX_VALUE;
        HashMap<Integer, Integer> window=new HashMap<>();
        HashMap<Integer, Integer> need=new HashMap<>();
        for (int s:small){
            need.put(s,need.getOrDefault(s,0)+1);
        }
        int match=0;
        while (right<big.length){
            int c1=big[right];
            if (need.containsKey(c1)){
            window.put(c1,window.getOrDefault(c1,0)+1);

                if (need.get(c1)==window.get(c1)){
                    match++;
                }
            }
            right++;
            while (match==need.size()){
                if (right-left<min){
                    start=left;
                    min=right-left;
                }
                int c2=big[left];
                if (need.containsKey(c2)){
                window.put(c2,window.getOrDefault(c2,0)-1);

                    if (need.get(c2)>window.get(c2)){
                        match--;
                    }
                }
                left++;
            }
        }
        return min== Integer.MAX_VALUE?new int[0]:new int[]{start,start+min-1};
    }

//    public int minimumTotal(List<List<Integer>> triangle) {
//        int n=triangle.size();
//        int[][] dp=new int[n][n];
//        dp[0][0]=triangle.get(0).get(0);
//        for(int i=1;i<n;i++){
//            dp[i][0]=triangle.get(i).get(0)+dp[i-1][0];
//            for(int j=1;j<i;j++){
//                dp[i][j]=Math.min(dp[i-1][j],dp[i-1][j-1])+triangle.get(i).get(j);
//            }
//            dp[i][i]=triangle.get(i).get(i)+dp[i-1][i-1];
//        }
//        Integer ans = Arrays.stream(dp[n - 1]).boxed().min(Comparator.naturalOrder()).orElse(null);
//        return ans;
//    }

    public int minimumTotal(List<List<Integer>> triangle) {
        int n=triangle.size();
        int[] dp=new int[n+1];
        for (int i=n-1;i>=0;i--){
            for (int j=0;j<=i;j++){
                dp[j]= Math.min(dp[j],dp[j+1])+triangle.get(i).get(j);
            }
        }
        return dp[0];
    }

    public int[] missingTwo(int[] nums) {
        int xor=0;
        for (int n:nums)xor^=n;
        int len=nums.length;
        for (int i=1;i<=(len+2);i++)xor^=i;
        int diff=xor&(-xor);
        int[] res=new int[2];
        for (int n:nums){
            if((n&diff)==0){
                res[0]^=n;
            }else res[1]^=n;
        }
        for (int n=1;n<=(len+2);n++){
            if((n&diff)==0){
                res[0]^=n;
            }else res[1]^=n;
        }
        return res;
    }

    List<String> wordList;
    boolean[] marked;
    List<String> output;
    String endWord;
    List<String> result;
    public List<String> findLadders(String beginWord, String endWord, List<String> wordList) {
        this.wordList=wordList;
        output=new ArrayList<>();
        marked=new boolean[wordList.size()];
        result=new ArrayList<>();
        this.endWord=endWord;
        dfs(beginWord);
        return result;
    }

    public void dfs(String s){
        output.add(s);
        Queue<String> q=oneCharDiff(s);
        for (String str:q){
            if (str.equals(endWord)){
                output.add(str);
                result=new ArrayList<>(output);
                return;
            }
            dfs(str);
            output.remove(output.size()-1);
        }
    }

    public Queue<String> oneCharDiff(String s){
        Queue<String> q=new LinkedList<>();
        for (int j=0;j<wordList.size();j++){
            String str=wordList.get(j);
            int diffNum=0;
            if (str.length()!=s.length()||marked[j])continue;
            for (int i=0;i<s.length();i++){
                if (diffNum>=2)break;
                if (str.charAt(i)!=s.charAt(i))diffNum++;
            }
            if (diffNum==1){
                q.add(str);
                marked[j]=true;
            }
        }
        return q;
    }

    public int numTrees(int n) {
        int[] dp=new int[n+1];
        dp[0]=1;
        dp[1]=1;
        for(int i=2;i<=n;i++){
            for (int j=1;j<=i;j++){
                dp[i]+=dp[j-1]*dp[i-j];
            }
        }
        return dp[n];
    }

    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map=new HashMap<>();
        for (int i=0;i<nums.length;i++){
            if (map.containsKey(target-nums[i])){
                return new int[]{i,map.get(target-nums[i])};
            }else {
                map.put(nums[i],i);
            }
        }
        return new int[0];
    }

     public class ListNode {
      int val;
      ListNode next;
      ListNode(int x) { val = x; }
     }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode ans=new ListNode(-1);
        int carry=0;
        ListNode cur=ans;
        while (l1!=null||l2!=null){
            int v1=l1==null?0:l1.val;
            int v2=l2==null?0:l2.val;
            int sum=v1+v2+carry;
            ListNode temp=new ListNode(sum%10);
            cur.next=temp;
            cur=temp;
            carry=sum/10;
            if (l1!=null)l1=l1.next;
            if (l2!=null)l2=l2.next;
        }
        if (carry!=0){
            cur.next=new ListNode(carry);
        }
        return ans.next;
    }

    public int lengthOfLongestSubstring(String s) {
        Set<Character> set=new HashSet<>();
        int max=0,left=0,right=0;
        while (left<s.length()&&right<s.length()){
            if (!set.contains(s.charAt(right))){
                set.add(s.charAt(right++));
                max= Math.max(max,right-left);
            }else {
                set.remove(s.charAt(left++));
            }
        }
        return max;
    }

    public boolean isBipartite(int[][] graph) {
        int n=graph.length;
        //0 uncolored
        //1 red
        //2 green
        int[] color=new int[n];
        for (int i=0;i<n;i++){
            if (color[i]==0){
                Queue<Integer> queue=new LinkedList<>();
                queue.offer(i);
                color[i]=1;
                while (!queue.isEmpty()){
                    int node=queue.poll();
                    int innerColor=color[node]==1?2:1;
                    for (int neighbor:graph[node]){
                        if (color[neighbor]==0){
                            queue.offer(neighbor);
                            color[neighbor]=innerColor;
                        }else if (color[neighbor]!=innerColor){
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }
}
//class LRUCache {
//    private int capacity;
//    private HashMap<Integer,Integer> map;
//    private LinkedList<Integer> list;
//    public LRUCache(int capacity) {
//        this.capacity=capacity;
//        map=new HashMap<>();
//        list=new LinkedList<>();
//    }
//
//    public int get(int key) {
//        if (map.containsKey(key)){
//            list.remove((Integer)key);
//            list.add(key);
//            return map.get(key);
//        }
//        return -1;
//    }
//
//    public void put(int key, int value) {
//        if (map.containsKey(key)){
//            list.remove((Integer)key);
//            list.add(key);
//            map.put(key,value);
//            return;
//        }
//        if (list.size()==capacity){
//            map.remove(list.removeFirst());
//            map.put(key,value);
//            list.add(key);
//        }else {
//            map.put(key,value);
//            list.add(key);
//        }
//    }
//
//
//}
