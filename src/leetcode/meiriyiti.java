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

    @Test
    public void t1(){
        LinkedList<Integer> a= new LinkedList<>();
        a.add(1);
        a.add(2);
        a.add(3);
        System.out.println(a.peek());
    }
}
