package packageProblem;

import java.util.Scanner;

public class packageMultiple {
    /**
     * f[i]体积j时最大总价值
     * 每件物品可以用si次
     * i:0->n  l:v[i]->m  f[j]=max(f[j],f[j-v[i]+w[i])
     * i:0->n  j:m->v[i] k:0->k*v[i]<=j  f[j]=max(f[j],f[j-k*v[i]+k*w[i])
     * O(n*n)
     *
     * @param args
     */
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();//个数
        int m = sc.nextInt();//容量
        int[] f = new int[m + 1];
        for (int i = 0; i < n; i++) {
            int v = sc.nextInt();
            int w = sc.nextInt();
            int s = sc.nextInt();
            for (int j = m; j >= 0; j--) {
                for (int k = 1; k <= s && k * v <= j; k++)
                    f[j] = Math.max(f[j], f[j - k * v] + k * w);
            }
        }
        System.out.println(f[m]);
    }
}
