package nowcoder.ProgrammerInterviewGuide;
/**题目描述
 给定一个二维数组map，含义是一张地图，例如，如下矩阵
 \begin{Bmatrix} -2&-3&3 \\ -5&-10&1\\ 0&30&-5 \end{Bmatrix}
 ⎩
 ⎨
 ⎧
 ​

 −2
 −5
 0
 ​

 −3
 −10
 30
 ​

 3
 1
 −5
 ​

 ⎭
 ⎬
 ⎫
 ​

 游戏的规则如下:
 1）骑士从左上角出发，每次只能向右或向下走，最后到达右下角见到公主。
 2）地图中每个位置的值代表骑士要遭遇的事情。如果是负数，说明此处有怪兽，要让骑士损失血量。如果是非负数，代表此处有血瓶，能让骑士回血。
 3）骑士从左上角到右下角的过程中，走到任何一个位置时，血量都不能少于1。为了保证骑土能见到公主，初始血量至少是多少?
 根据map,输出初始血量。

 输入描述:
 第一行两个正整数n，m  \left ( 1\leq n,m\leq 10^{3} \right )(1≤n,m≤10
 3
 )，接下来n行，每行m个整数，代表map_{ij} \left( -10^3 \leq map_{ij} \leq 10^{3}\right )map
 ij
 ​
 (−10
 3
 ≤map
 ij
 ​
 ≤10
 3
 )。
 输出描述:
 输出一个整数，表示答案。
 示例1
 输入
 3 3
 -2 -3 3
 -5 -10 1
 0 30 -5
 输出
 7
 *
 */

import java.util.Scanner;

public class CD45 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int m= scanner.nextInt();
        int[][] matrix=new int[n][m];
        for (int i=0;i<n;i++){
            for (int j=0;j<m;j++){
                matrix[i][j]= scanner.nextInt();
            }
        }
        int[][] dp=new int[n][m];
        if (matrix[n-1][m-1]<=0){
            dp[n-1][m-1]=1-matrix[n-1][m-1];
        }else {
            dp[n-1][m-1]=1;
        }
        for (int j=m-2;j>=0;j--){
            if (dp[n-1][j+1]-matrix[n-1][j]>0){
                dp[n-1][j]=dp[n-1][j+1]-matrix[n-1][j];
            }else {
                dp[n-1][j]=1;
            }
        }
        for (int i=n-2;i>=0;i--){
            if (dp[i+1][m-1]-matrix[i][m-1]>0){
                dp[i][m-1]=dp[i+1][m-1]-matrix[i][m-1];
            }else {
                dp[i][m-1]=1;
            }
        }
        for (int i=n-2;i>=0;i--){
            for (int j=m-2;j>=0;j--){
                int right=dp[i][j+1]-matrix[i][j]>0?dp[i][j+1]-matrix[i][j]:1;
                int down=dp[i+1][j]-matrix[i][j]>0?dp[i+1][j]-matrix[i][j]:1;
                dp[i][j]=Math.min(right,down);
            }
        }
        System.out.println(dp[0][0]);
    }
}
