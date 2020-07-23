package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定数组arr，arr中所有的值都为正整数且不重复。每个值代表一种面值的货币，每种面值的货币可以使用任意张，再给定一个aim，代表要找的钱数，求组成aim的最少货币数。
 * 输入描述:
 * 输入包括两行，第一行两个整数n（0<=n<=1000）代表数组长度和aim（0<=aim<=5000），第二行n个不重复的正整数，代表arr\left( 1 \leq arr_i \leq 10^9 \right)(1≤arr
 * i
 * ​
 *  ≤10
 * 9
 *  )。
 * 输出描述:
 * 输出一个整数，表示组成aim的最小货币数，无解时输出-1.
 * 示例1
 * 输入
 * 3 20
 * 5 2 3
 * 输出
 * 4
 * 说明
 * 20=5*4
 */

import java.util.Arrays;
import java.util.Scanner;

public class CD12 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int target=scanner.nextInt();
        long[] dp=new long[target+1];
        Arrays.fill(dp,Integer.MAX_VALUE);
        dp[0]=0;
        for (int i=0;i<n;i++){
            int x=scanner.nextInt();
            for (int j=x;j<=target;j++){
                dp[j]=Math.min(dp[j],dp[j-x]+1);
            }
        }
        System.out.println(dp[target]==Integer.MAX_VALUE?-1:dp[target]);
    }
}
