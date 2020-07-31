package leetcode;

import com.sun.tracing.dtrace.ArgsAttributes;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

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
}
