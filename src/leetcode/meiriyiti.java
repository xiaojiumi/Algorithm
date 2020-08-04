package leetcode;

import com.sun.tracing.dtrace.ArgsAttributes;
import org.junit.Test;
import sun.reflect.generics.tree.Tree;

import java.util.*;

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
}
