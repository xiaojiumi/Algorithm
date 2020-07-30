package nowcoder.ProgrammerInterviewGuide;
/**题目描述
 给定一个N \times NN×N的矩阵matrix，在这个矩阵中，只有0和1两种值，返回边框全是1的最大正方形的边长长度、
 例如
 0 1 1 1 1
 0 1 0 0 1
 0 1 0 0 1
 0 1 1 1 1
 0 1 0 1 1
 其中，边框全是1的最大正方形的大小为4 ×4，所以返回4
 [要求]时间复杂度为O(n^3)，空间复杂度为O(n^2)
 输入描述:
 第一行一个整数N。表示矩阵的长宽。
 接下来N行，每行N个整数表示矩阵内的元素
 输出描述:
 输出一个整数表示答案
 示例1
 输入
 5
 0 1 1 1 1
 0 1 0 0 1
 0 1 0 0 1
 0 1 1 1 1
 0 1 0 1 1
 输出
 4
 *
 */

import java.util.Scanner;

public class CD41 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int[][] nums=new int[n][n];
        for (int i=0;i<n;i++){
            for (int j=0;j<n;j++){
                nums[i][j]= scanner.nextInt();
            }
        }
        int[][] right=new int[n][n];
        int[][] down=new int[n][n];
        for (int i=n-1;i>=0;i--){
            for (int j=n-1;j>=0;j--){
                if (i==n-1){
                    down[i][j]=nums[i][j];
                }else {
                    down[i][j]=nums[i][j]==0?0:down[i+1][j]+1;
                }
                if (j==n-1){
                    right[i][j]=nums[i][j];
                }else {
                    right[i][j]=nums[i][j]==0?0:right[i][j+1]+1;
                }
            }
        }
        int max=0;
        for (int i=0;i<n;i++){
            for (int j=0;j<n;j++){
                int m=Math.min(down[i][j],right[i][j]);
                for (int k=m-1;k>=0;k--){
                    if (right[i+k][j]>=k+1&&down[i][j+k]>=k+1){
                        max=Math.max(max,k+1);
                        break;
                    }
                }
            }
        }
        System.out.println(max);
    }
}
