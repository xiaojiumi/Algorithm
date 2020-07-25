package nowcoder.ProgrammerInterviewGuide;
/**
 * 输入描述:
 * 输出包括两行，第一行代表字符串str1，第二行代表str2。\left( 1\leq length(str1),length(str2) \leq 5000\right)(1≤length(str1),length(str2)≤5000)
 * 输出描述:
 * 输出一行，代表他们最长公共子序列。如果公共子序列的长度为空，则输出-1。
 * 示例1
 * 输入
 * 1A2C3D4B56
 * B1D23CA45B6A
 * 输出
 * 123456
 * 说明
 * "123456"和“12C4B6”都是最长公共子序列，任意输出一个。
 */

import java.util.Scanner;

public class CD31 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String s1 = scanner.nextLine();
        String s2 = scanner.nextLine();
        int n=s1.length(),m=s2.length();
        int[][] dp=new int[n+1][m+1];
        for (int i=1;i<=n;i++){
            for (int j=1;j<=m;j++){
                if (s1.charAt(i-1)==s2.charAt(j-1))
                    dp[i][j]=dp[i-1][j-1]+1;
                else dp[i][j]=Math.max(dp[i][j-1],dp[i-1][j]);
            }
        }
        StringBuilder sb=new StringBuilder();
        int index=0;
        int i=n,j=m;
        while (index<dp[n][m]){
            if (s1.charAt(i-1)==s2.charAt(j-1)){
                sb.append(s1.charAt(i-1));
                i--;
                j--;
                index++;
            }else {
                if (dp[i][j - 1] > dp[i - 1][j]) {
                    j--;
                }else {
                    i--;
                }
            }
        }
        System.out.println(sb.reverse().toString());
    }
}
