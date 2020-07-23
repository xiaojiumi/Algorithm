package packageProblem;

import java.util.Scanner;

public class package01 {
    /**
     * f[i][j]前i个物品，体积j，总价值
     * 每件物品只能用一次
     * f[i][j]=max(f[i-1][j],f[i][j-v[i]]+w[i])
     * O(n*n)
     *
     * @param args
     */
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();//个数
        int m = sc.nextInt();//容量
        int[] v = new int[n];
        int[] w = new int[n];
        for (int i = 0; i < n; i++) {
            v[i] = sc.nextInt();
            w[i] = sc.nextInt();
        }
        int[] f = new int[m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = m; j >= v[i - 1]; j--) {
                f[j] = Math.max(f[j], f[j - v[i - 1]] + w[i - 1]);
            }
        }
        System.out.println(f[m]);
    }
}
