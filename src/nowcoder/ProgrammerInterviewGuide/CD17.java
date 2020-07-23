package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 假设有排成一行的N个位置，记为1~N，开始时机器人在M位置，机器人可以往左或者往右走，如果机器人在1位置，那么下一步机器人只能走到2位置，如果机器人在N位置，那么下一步机器人只能走到N-1位置。规定机器人只能走k步，最终能来到P位置的方法有多少种。由于方案数可能比较大，所以答案需要对1e9+7取模。
 * 输入描述:
 * 输出包括一行四个正整数N（2<=N<=5000）、M(1<=M<=N)、K(1<=K<=5000)、P(1<=P<=N)。
 * 输出描述:
 * 输出一个整数，代表最终走到P的方法数对10^9+710
 * 9
 *  +7取模后的值。
 * 示例1
 * 输入
 * 5 2 3 3
 * 输出
 * 3
 * 说明
 * 1).2->1,1->2,2->3
 *
 * 2).2->3,3->2,2->3
 *
 * 3).2->3,3->4,4->3
 */

import java.util.Scanner;

public class CD17 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int m=scanner.nextInt();
        int k= scanner.nextInt();
        int p= scanner.nextInt();
        int[][] dp=new int[k+1][n+1];
        dp[0][p]=1;
        int mod=1000000007;
        for (int i=1;i<=k;i++){
            for (int j=1;j<=n;j++){
                if (j==1){
                    dp[i][j]=dp[i-1][j+1]%mod;
                }else if (j==n){
                    dp[i][j]=dp[i-1][j-1]%mod;
                }else dp[i][j]=dp[i-1][j-1]%mod+dp[i-1][j+1]%mod;
            }
        }
        System.out.println(dp[k][m]%mod);
    }
}
