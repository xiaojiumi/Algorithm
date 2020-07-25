package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定数组arr，设数组长度为n，arr中所有的值都为正整数且不重复。每个值代表一种面值的货币，每种面值的货币可以使用任意张，再给定一个整数aim，代表要找的钱数，求换钱的方法数有多少种。由于方法的种数比较大，所以要求输出对10^9+710
 * 9
 *  +7进行取模后的答案。
 * 输入描述:
 * 输出包括两行，第一行包括两个整数n(0 \leq n \leq 1000)(0≤n≤1000)和aim(0 \leq aim \leq 20000)(0≤aim≤20000)。第二行包含n个整数，表示arr数组\left( 1 \leq arr_i \leq 1e9 \right)(1≤arr
 * i
 * ​
 *  ≤1e9)。
 * 输出描述:
 * 输出一个整数，表示换钱的方法数对10^9+710
 * 9
 *  +7取模后的答案。
 * 示例1
 * 输入
 * 4 15
 * 5 10 25 1
 * 输出
 * 6
 * 说明
 * 5*3=15
 *
 * 10*1+5*1=15
 *
 * 10*1+1*5=15
 *
 * 1*10+5*1=15
 *
 * 5*2+1*5=15
 *
 * 1*15=15
 */

import java.util.Scanner;

public class CD19 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int aim=scanner.nextInt();
        int[] nums=new int[n];
        for(int i=0;i<n;i++){
            nums[i] = scanner.nextInt();
        }
        int[] dp=new int[aim+1];
        dp[0]=1;
        for (int i=0;i<n;i++){
            for (int j=nums[i];j<=aim;j++){
                dp[j]= (dp[j-nums[i]]+dp[j])%1_0000_0000_7;
            }
        }
        System.out.println(dp[aim]);
    }
}
