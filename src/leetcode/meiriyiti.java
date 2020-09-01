package leetcode;

import org.junit.Test;

import java.util.*;
import java.util.stream.Collectors;

public class meiriyiti {

    public List<TreeNode> generateTrees(int n) {
        if (n==0)return new ArrayList<>();
        return backtrack(1,n);
    }

    public List<TreeNode> backtrack(int start,int end){
        List<TreeNode> ans=new ArrayList<>();
        if(start>end){
            ans.add(null);
            return ans;
        }
        for (int i=start;i<=end;i++){
            List<TreeNode> left=backtrack(start,i-1);
            List<TreeNode> right=backtrack(i+1,end);
            for (TreeNode l:left){
                for (TreeNode r:right){
                    TreeNode cur=new TreeNode(i);
                    cur.left=l;
                    cur.right=r;
                    ans.add(cur);
                }
            }
        }
        return ans;
    }

    public int minArray(int[] numbers) {
        int i=0,j=numbers.length-1;
        while (i<j){
            int m=(i+j)>>1;
            if (numbers[m]>numbers[j])i=m+1;
            else if (numbers[m]<numbers[j])j=m;
            else j--;
        }
        return numbers[i];
    }

    public int minPathSum(int[][] grid) {
        int n=grid.length,m=grid[0].length;
        int[][] dp=new int[n][m];
        dp[0][0]=grid[0][0];
        for (int i=1;i<n;i++)dp[i][0]=grid[i][0]+dp[i-1][0];
        for (int i=1;i<m;i++)dp[0][i]=grid[0][i]+dp[0][i-1];
        for (int i=1;i<n;i++){
            for (int j=1;j<m;j++){
                dp[i][j]=Math.min(dp[i][j-1],dp[i-1][j])+grid[i][j];
            }
        }
        return dp[n-1][m-1];
    }

    public boolean divisorGame(int N) {
        boolean[] dp=new boolean[N+1];
        dp[1]=false;
        dp[2]=true;
        for (int i=3;i<=N;i++){
            for (int j=1;j<i;j++){
                if ((i%j)==0&&!dp[i-j]){
                    dp[i]=true;
                    break;
                }
            }
        }
        return dp[N];
    }

    public int splitArray(int[] nums, int m) {
        int n = nums.length;
        int[][] f = new int[n + 1][m + 1];
        for (int i = 0; i <= n; i++) {
            Arrays.fill(f[i], Integer.MAX_VALUE);
        }
        int[] sub = new int[n + 1];
        for (int i = 0; i < n; i++) {
            sub[i + 1] = sub[i] + nums[i];
        }
        f[0][0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= Math.min(i, m); j++) {
                for (int k = 0; k < i; k++) {
                    f[i][j] = Math.min(f[i][j], Math.max(f[k][j - 1], sub[i] - sub[k]));
                }
            }
        }

        return f[n][m];
    }

    public int[][] dirs={{-1,0},{1,0},{0,-1},{0,1}};
    public int rows,columns;
    public int longestIncreasingPath(int[][] matrix) {
        if (matrix==null||matrix.length==0||matrix[0].length==0){
            return 0;
        }
        rows=matrix.length;
        columns=matrix[0].length;
        int[][] memo=new int[rows][columns];
        int ans=0;
        for (int i=0;i<rows;i++){
            for (int j=0;j<columns;j++){
                ans=Math.max(ans,dfs(matrix,i,j,memo));
            }
        }
        return ans;
    }

    public int dfs(int[][] matrix,int row,int column,int[][] memo){
        if (memo[row][column]!=0){
            return memo[row][column];
        }
        memo[row][column]++;
        for (int[] dir:dirs){
            int newRow=row+dir[0],newColumn=column+dir[1];
            if (newRow>=0&&newRow<rows
            &&newColumn>=0&&newColumn<columns
            &&matrix[newRow][newColumn]>matrix[row][column]){
                memo[row][column]=Math.max(memo[row][column],dfs(matrix,newRow,newColumn,memo)+1);
            }
        }
        return memo[row][column];
    }

    public boolean isSubsequence(String s, String t) {
        int index=-1;
        for (char c:s.toCharArray()){
            index=t.indexOf(c,index+1);
            if (index==-1)return false;
        }
        return true;
    }

    public int integerBreak(int n) {
        int[] dp=new int[n+1];
        for (int i=2;i<=n;i++){
            int temp=0;
            for (int j=1;j<i;j++){
                temp=Math.max(temp,Math.max(j*(i-j),j*dp[i-j]));
            }
            dp[i]=temp;
        }
        return dp[n];
    }

    public int maxDepth(TreeNode root) {
        if(root==null)return 0;
        return Math.max(maxDepth(root.left),maxDepth(root.right))+1;
    }

    int magicIndex=-1;
    public int findMagicIndex(int[] nums) {
        search(nums,0,nums.length-1);
        return magicIndex;
    }

    public void search(int[] nums,int left,int right){
        if (left>right)return ;
        int mid=(left+right)/2;
        if (nums[mid]==mid){
            magicIndex=mid;
            search(nums,left,mid-1);
        }else {
            search(nums,left,mid-1);
            if (magicIndex==-1)search(nums,mid+1,right);
        }
    }

//    public void flatten(TreeNode root) {
//        List<TreeNode> list=new ArrayList<>();
//        preOrder(root,list);
//        int size=list.size();
//        for (int i=1;i<size;i++){
//            TreeNode prev=list.get(i-1);
//            TreeNode cur=list.get(i);
//            prev.left=null;
//            prev.right=cur;
//        }
//    }
//
//    public void preOrder(TreeNode root,List<TreeNode> list){
//        if (root!=null){
//            list.add(root);
//            preOrder(root.left,list);
//            preOrder(root.right,list);
//        }
//    }

    public void flatten(TreeNode root){
        while (root!=null){
            if (root.left != null) {
                TreeNode pre = root.left;
                while (pre.right != null) {
                    pre = pre.right;
                }
                pre.right = root.right;
                root.right = root.left;
                root.left = null;
            }
            root=root.right;
        }
    }

    public String addStrings(String num1, String num2) {
        int n=num1.length(),m=num2.length();
        StringBuffer sb=new StringBuffer();
        int i=n-1,j=m-1;
        int carry=0;
        while (i>=0||j>=0){
            int a=i>=0? num1.charAt(i)-'0' :0;
            int b=j>=0?num2.charAt(j)-'0':0;
            int sum=a+b+carry;
            sb.append(sum%10);
            carry=sum/10;
            i--;
            j--;
        }
        if (carry!=0)sb.append(carry);
        return sb.reverse().toString();
    }

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] indegrees=new int[numCourses];
        List<List<Integer>> adjacency=new ArrayList<>();
        Queue<Integer> queue=new ArrayDeque<>();
        for (int i=0;i<numCourses;i++)adjacency.add(new ArrayList<>());
        for (int[] p:prerequisites){
            indegrees[p[0]]++;
            adjacency.get(p[1]).add(p[0]);
        }
        for (int i=0;i<numCourses;i++){
            if (indegrees[i]==0)queue.add(i);
        }
        while (!queue.isEmpty()){
            int pre=queue.poll();
            numCourses--;
            for (int cur:adjacency.get(pre)){
                indegrees[cur]--;
                if (indegrees[cur]==0)queue.add(cur);
            }
        }
        return numCourses==0;
    }

    public int rob(TreeNode root) {
        int[] result=robHelp(root);
        return Math.max(result[0],result[1]);
    }

    public int[] robHelp(TreeNode root){
        if (root==null)return new int[2];
        int[] result=new int[2];
        int[] left=robHelp(root.left);
        int[] right=robHelp(root.right);
        result[0]=Math.max(left[0], left[1] )+Math.max(right[0], right[1] );
        result[1]=left[0]+ right[0]+root.val;
        return result;
    }


    public List<String> restoreIpAddresses(String s) {
        List<String> result=new ArrayList<>();
        StringBuilder ip=new StringBuilder();
        for (int a=1;a<4;a++){
            for (int b=1;b<4;b++){
                for (int c=1;c<4;c++){
                    for (int d=1;d<4;d++){
                        if (a+b+c+d==s.length()){
                            int seg1=Integer.parseInt(s.substring(0,a));
                            int seg2 = Integer.parseInt(s.substring(a, a + b));
                            int seg3 = Integer.parseInt(s.substring(a + b, a + b + c));
                            int seg4 = Integer.parseInt(s.substring(a + b + c, a + b + c + d));
                            if (seg1<=255&&seg2<=255&&seg3<=255&&seg4<=255){
                                ip.append(seg1).append(".").append(seg2).append(".").
                                        append(seg3).append(".").append(seg4);
                                if (ip.length()==s.length()+3)result.add(ip.toString());
                                ip.delete(0,ip.length());
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    public int countBinarySubstrings(String s) {
        int[] array=new int[s.length()];
        array[0]=1;
        int index=0;
        for (int i=1;i<s.length();i++){
            if (s.charAt(i)==s.charAt(i-1)){
                array[index]++;
            }else {
                array[++index]=1;
            }
        }
        int ans=0;
        for (int i=1;i<array.length;i++){
            ans+=Math.min(array[i-1],array[i]);
        }
        return ans;
    }

    public void solve(char[][] board) {
        if (board==null||board.length==0)return;
        int n=board.length,m=board[0].length;
        for (int i=0;i<n;i++){
            if (board[i][0]=='O'){
                dfs(board,i,0);
            }
            if (board[i][m-1]=='O'){
                dfs(board,i,m-1);
            }
        }
        for (int i=0;i<m;i++){
            if (board[0][i]=='O'){
                dfs(board,0,i);
            }
            if (board[n-1][i]=='O'){
                dfs(board,n-1,i);
            }
        }
        for (int i=0;i<n;i++){
           for (int j=0;j<m;j++){
               if (board[i][j]!='A')board[i][j]='X';
               else board[i][j]='O';
           }
        }
    }

    public void dfs(char[][] board,int row,int col){
        if (row<0||row>=board.length||col<0||col>=board[0].length||board[row][col]!='O')return;
        board[row][col]='A';
        dfs(board,row+1,col);
        dfs(board,row-1,col);
        dfs(board,row,col+1);
        dfs(board,row,col-1);
    }

    private HashMap<Node,Node> visited=new HashMap<>();
    public Node cloneGraph(Node node) {
        if(node==null)return node;
        if (visited.containsKey(node))return visited.get(node);
        Node cloneNode=new Node(node.val,new ArrayList<>());
        visited.put(node,cloneNode);
        for (Node neigh: node.neighbors){
            cloneNode.neighbors.add(cloneGraph(neigh));
        }
        return cloneNode;
    }

    @Test
    public void t1(){
        meiriyiti m=new meiriyiti();
        System.out.println(12%10);
    }

    public String multiply(String num1, String num2) {
        if(num1.equals("0")||num2.equals("0"))return "0";
        int n=num1.length(),m=num2.length();
        int[] nums=new int[n+m];
        for (int i=n-1;i>=0;i--){
            int x = num1.charAt(i) - '0';
            for (int j=m-1;j>=0;j--){
                int y = num2.charAt(j) - '0';
                nums[i+j+1]+=x*y;
            }
        }
        for (int i=m+n-1;i>0;i--){
            nums[i-1]+=nums[i]/10;
            nums[i]%=10;
        }
        StringBuilder sb=new StringBuilder();
        int index=nums[0]==0?1:0;
        while (index<m+n){
            sb.append(nums[index]);
            index++;
        }
        return sb.toString();
    }

    public boolean isValid(String s) {
        Stack<Character> stack=new Stack<>();
        for (char c:s.toCharArray()){
            if (c=='(')stack.push(')');
            else if (c=='[')stack.push(']');
            else if (c=='{')stack.push('}');
            else if (stack.isEmpty()||c!=stack.pop())return false;
        }
        return stack.isEmpty();
    }

    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int start= image[sr][sc];
        if (start==newColor)return image;
        int[] dx={1,-1,0,0};
        int[] dy={0,0,1,-1};
        Queue<int[]> queue=new LinkedList<>();
        queue.add(new int[]{sr,sc});
        while (!queue.isEmpty()){
            int[] poll = queue.poll();
            int x= poll[0],y=poll[1];
            image[x][y]=newColor;
            for (int i=0;i<4;i++){
                int newX=x+dx[i],newY=y+dy[i];
                if (newX>=0&&newX<image.length&&newY>=0&&newY<image[0].length
                &&image[newX][newY]==start){
                    queue.add(new int[]{newX,newY});
                }
            }
        }
        return image;
    }

    boolean ans=true;
    public boolean isBalanced(TreeNode root) {
        getHeight(root);
        return ans;
    }

    public int getHeight(TreeNode root){
        if (root==null)return 0;
        int l=getHeight(root.left);
        int r=getHeight(root.right);
        if (Math.abs(l-r)>1)ans=false;
        return Math.max(l,r)+1;
    }



//    public TreeNode sortedListToBST(ListNode head) {
//        return build(head,null);
//    }
//
//    public TreeNode build(ListNode left,ListNode right){
//        if(left==right)return null;
//        ListNode mid=getMid(left,right);
//        TreeNode root = new TreeNode(mid.val);
//        root.left=build(left,mid);
//        root.right=build(mid.next,right);
//        return root;
//    }
//
//    public ListNode getMid(ListNode left,ListNode right){
//        ListNode fast=left,slow=left;
//        while (fast!=right&&fast.next!=right){
//            fast=fast.next.next;
//            slow=slow.next;
//        }
//        return slow;
//    }
    public TreeNode sortedListToBST(ListNode head) {
        ArrayList<Integer> list = new ArrayList<>();
        while (head!=null){
            list.add(head.val);
            head=head.next;
        }
        return getTree(list,0,list.size()-1);
    }

    public TreeNode getTree(ArrayList<Integer> list,int start,int end){
        if (start>end)return null;
        int mid=(start+end)/2;
        TreeNode root=new TreeNode(list.get(mid));
        root.left=getTree(list,start,mid-1);
        root.right=getTree(list,mid+1,end);
        return root;
    }

    public int countSubstrings(String s) {
        int ans=0;
        for (int i=0;i<s.length();i++){
            ans+=helper(s,i,i);
            ans+=helper(s,i,i+1);
        }
        return ans;
    }

    public int helper(String s,int start,int end){
        int temp=0;
        while (start>=0&&end<s.length()&&s.charAt(start)==s.charAt(end)){
            start--;
            end++;
            temp++;
        }
        return temp;
    }

    int[] dirX={0,0,1,-1,1,1,-1,-1};
    int[] dirY ={1,-1,0,0,1,-1,1,-1};
    public char[][] updateBoard(char[][] board, int[] click) {
        int x=click[0],y= click[1];
        if (board[x][y]=='M'){
            board[x][y]='X';
        }else {
            dfsBoard(board,x,y);
        }
        return board;
    }

    public void dfsBoard(char[][] board,int x,int y){
        int count=0;
        for (int i=0;i<8;i++){
            int tx=x+dirX[i],ty=y+ dirY[i];
            if (tx<0||tx>= board.length||ty<0||ty>= board[0].length)continue;
            if (board[tx][ty]=='M'){
                count++;
            }
        }
        if (count>0){
            board[x][y]=(char)(count+'0');
        }else {
            board[x][y]='B';
            for (int i=0;i<8;i++){
                int tx=x+dirX[i],ty=y+ dirY[i];
                if (tx<0||tx>= board.length||
                    ty<0||ty>= board[0].length||
                    board[tx][ty]!='E')continue;
                dfsBoard(board,tx,ty);
            }
        }
    }

    public int minDepth(TreeNode root) {
        if(root==null)return 0;
        if (root.left==null&&root.right==null)return 1;
        int res=Integer.MAX_VALUE;
        if (root.left!=null)res=Math.min(res,minDepth(root.left));
        if (root.right!=null)res=Math.min(res,minDepth(root.right));
        return res+1;
    }

    public boolean judgePoint24(int[] nums) {
        ArrayList<Double> doubles = new ArrayList<>();
        for (int a:nums){
            doubles.add((double) a);
        }
        return helper(doubles);
    }

    public boolean helper(ArrayList<Double> arr){
        if (arr.size()==0)return false;
        if(arr.size()==1){
            return Math.abs(arr.get(0)-24)<1e-6;
        }
        for (int i=0;i<arr.size();i++){
            for (int j=0;j<arr.size();j++){
                if(i!=j){
                    ArrayList<Double> temp=new ArrayList<>();
                    for (int k=0;k<arr.size();k++){
                        if (k!=i&&k!=j){
                            temp.add(arr.get(k));
                        }
                    }
                    for (int k=0;k<4;k++){
                        if (k==0)temp.add(arr.get(i)+arr.get(j));
                        if (k==1)temp.add(arr.get(i)-arr.get(j));
                        if (k==2)temp.add(arr.get(i)*arr.get(j));
                        if (k==3){
                            if (arr.get(j)!=0){
                                temp.add(arr.get(i)/arr.get(j));
                            }else {
                                continue;
                            }
                        }
                        if(helper(temp))return true;
                        temp.remove(temp.size()-1);
                    }
                }
            }
        }
        return false;
    }

    public int rangeBitwiseAnd(int m, int n) {
         while (m!=n){
             n=n&(n-1);
         }
         return n;
    }

//    public boolean repeatedSubstringPattern(String s) {
//        int n=s.length();
//        for(int i=1;i<=n/2;i++){
//            if (n%i==0){
//                boolean flag=true;
//                for (int j=i;j<n;j++){
//                    if (s.charAt(j)!=s.charAt(j-i)){
//                        flag=false;
//                        break;
//                    }
//                }
//                if(flag)return true;
//            }
//        }
//        return false;
//    }

    public boolean repeatedSubstringPattern(String s) {
        return (s+s).indexOf(s,1)!=s.length();
    }


    List<Integer> temp = new ArrayList<Integer>();
    List<List<Integer>> res = new ArrayList<List<Integer>>();
    Set<Integer> set = new HashSet<Integer>();
    int n;
    public List<List<Integer>> findSubsequences(int[] nums) {
        n= nums.length;;
        for (int i=0;i<( 1<<n);i++){
            findSubsequences(i,nums);
            int hashValue = getHash(263, (int) 1E9 + 7);
            if (check()&&!set.contains(hashValue)){
                set.add(hashValue);
                res.add(new ArrayList<>(temp));
            }
        }
        return res;
    }

    public void findSubsequences(int mask, int[] nums) {
        temp.clear();
        for (int i = 0; i < n; ++i) {
            if ((mask & 1) != 0) {
                temp.add(nums[i]);
            }
            mask >>= 1;
        }
    }

    public int getHash(int base, int mod) {
        int hashValue = 0;
        for (int x : temp) {
            hashValue = hashValue * base % mod + (x + 101);
            hashValue %= mod;
        }
        return hashValue;
    }

    public boolean check() {
        for (int i = 1; i < temp.size(); ++i) {
            if (temp.get(i) < temp.get(i - 1)) {
                return false;
            }
        }
        return temp.size() >= 2;
    }

    public List<String> letterCombinations(String digits) {
        if (digits==null||digits.length()==0)return new ArrayList<>();
        List<String> ans=new ArrayList<>();
        String[] data={"abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
        backtrack(digits,ans,new StringBuilder(),0,data);
        return ans;
    }

    public void backtrack(String digits,List<String> ans,
                          StringBuilder cur,int index,
                          String[] data){
        if (cur.length()== digits.length()){
            ans.add(cur.toString());
            return;
        }
        String s=data[digits.charAt(index)-'2'];
        for (char c:s.toCharArray()){
            cur.append(c);
            backtrack(digits,ans,cur,index+1,data);
            cur.deleteCharAt(cur.length()-1);
        }
    }

    public List<String> findItinerary(List<List<String>> tickets) {
        Map<String,PriorityQueue<String>> map=new HashMap<>();
        List<String> ans=new ArrayList<>();
        for(List<String> ticket:tickets){
            String src=ticket.get(0),det=ticket.get(1);
            if (!map.containsKey(src)){
                map.put(src,new PriorityQueue<>());
            }
            map.get(src).offer(det);
        }
        dfs("JFK",map,ans);
        Collections.reverse(ans);
        return ans;
    }
    public void dfs(String cur,Map<String,PriorityQueue<String>> map, List<String> ans){
        while (map.containsKey(cur)&&map.get(cur).size()>0){
            String temp=map.get(cur).poll();
            dfs(temp,map,ans);
        }
        ans.add(cur);
    }

    public boolean judgeCircle(String moves) {
        int x=0,y=0;
        for (char c:moves.toCharArray()){
            if (c=='U')x++;
            if (c=='D')x--;
            if(c=='L')y--;
            if (c=='R')y++;
        }
        return x==0&&y==0;
    }

    public String reverseWords(String s) {
        return Arrays.stream(s.split(" ")).map(o->new StringBuffer(o).reverse().toString()).collect(Collectors.joining(" "));
    }

    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        Queue<Integer> queue=new LinkedList<>();
        for (int i:rooms.get(0)){
            queue.add(i);
        }
        Set<Integer> set=new HashSet<>();
        boolean[] visited=new boolean[rooms.size()];
        visited[0]=true;
        set.add(0);
        while (!queue.isEmpty()){
            int temp=queue.poll();
            if (!visited[temp]){
                set.add(temp);
                visited[temp]=true;
                for (int i:rooms.get(temp)){
                    queue.add(i);
                }
            }
        }
        return set.size()==rooms.size();
    }

    public boolean PredictTheWinner(int[] nums) {
        return total(nums,0, nums.length-1, 1)>=0;
    }

    public int total(int[] nums,int start,int end,int turn){
        if (start==end){
            return nums[start]*turn;
        }
        int scoreStart=nums[start]*turn+total(nums,start+1,end,-turn);
        int scoreEnd=nums[end]*turn+total(nums,start,end-1,-turn);
        if (turn==1)return Math.max(scoreStart,scoreEnd);
        else return Math.min(scoreStart,scoreEnd);
    }
}
