package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定一个字符串str，str全部由数字字符组成，如果str中的某一个或者相邻两个字符组成的子串值在1~26之间，则这个子串可以转换为一个字母。规定‘1’转换为“A”，“2”转换为“B”......"26"转化为“Z“。请求出str有多少种不同的转换结果，由于答案可能会比较大，所以请输出对10^{9}+710
 * 9
 *  +7取模后的答案。
 * 输入描述:
 * 输出一行仅有’0‘~’9‘组成的字符串，代表str \left(  0\leq length\left( str\right) \leq 100000 \right)(0≤length(str)≤100000)。
 * 输出描述:
 * 输出一个整数，代表你所求出的取模后答案。
 * 示例1
 * 输入
 * 1111
 * 输出
 * 5
 * 说明
 * 能转换出来的结果有：“AAAAA”，“LAA”，“ALA”，“AAL”，“LL”。
 */

import java.util.Scanner;

public class CD46 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String s= scanner.nextLine();
        int n=s.length();
        long[] dp=new long[n+1];
        long mod = 1000000007;
        if (n==0||s.charAt(0)=='0') {
            System.out.println(0);
            return;
        }
        dp[0]=dp[1]=1;
        for (int i=2;i<=n;i++){
            if (s.charAt(i-1)!='0'){
                dp[i]=(dp[i-1]+dp[i])%mod;
            }
            String temp=s.substring(i-2,i);
            Integer value = Integer.valueOf(temp);
            if (value>=10&&value<=26){
                dp[i]=(dp[i-2]+dp[i])%mod;
            }

        }
        System.out.println(dp[n]);
    }
}
