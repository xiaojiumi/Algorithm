package nowcoder.ProgrammerInterviewGuide;

import java.util.Scanner;

public class CD1 {

    /**
     * 题目描述
     * 给定一个N \times MN×M的整形矩阵matrix和一个整数K, matrix的每一行和每一列都是排好序的。
     * 实现一个函数，判断K是否在matrix中
     * [要求]
     * 时间复杂度为O(N+M)O(N+M)，额外空间复杂度为O(1)O(1)。
     * 输入描述:
     * 第一行有三个整数N, M, K
     * 接下来N行，每行M个整数为输入的矩阵
     * 输出描述:
     * 若K存在于矩阵中输出"Yes"，否则输出"No"
     *
     * 输入
     * 2 4 5
     * 1 2 3 4
     * 2 4 5 6
     * 输出
     * Yes
     * @param args
     */
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int m=scanner.nextInt();
        int k=scanner.nextInt();
        int[][] data=new int[n][m];
        for (int i=0;i<n*m;i++){
            data[i/m][i%m]=scanner.nextInt();
        }
        int row=0,col=m-1;
        while (row>=0&&row<n&&col>=0&&col<m){
            int temp=data[row][col];
            if (temp==k) {
                System.out.println("Yes");
                return;
            }else if (temp>k){
                col--;
            }else {
                row++;
            }
        }
        System.out.println("No");
    }
}
