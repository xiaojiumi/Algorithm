package packageProblem;

import java.util.ArrayList;
import java.util.Scanner;

public class packageMultipleFinal {
    /**
     * 多重背包使用单调队列优化
     * f[j]=f[j-v]+w,f[j-2*v]+2*w,...f[j-k*v]+k*w
     * f[j+v]=f[j]+w,f[j-v]+2*w
     * 每次求最大值时值变化了，但是值之间的大小关系没有变化（都增加了w）
     *
     * @param args
     */
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();//个数
        int m = sc.nextInt();//容量
        int[] f = new int[m + 1];
        ArrayList<Good> goods = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int v = sc.nextInt();
            int w = sc.nextInt();
            int s = sc.nextInt();
            for (int k = 1; k <= s; k *= 2) {
                s -= k;
                goods.add(new Good(v * k, w * k));
            }
            if (s > 0) goods.add(new Good(v * s, w * s));
        }
        for (Good good : goods)
            for (int j = m; j >= good.v; j--)
                f[j] = Math.max(f[j], f[j - good.v] + good.w);
        System.out.println(f[m]);
    }

    static class Good {
        int v;//体积
        int w;//价值

        public int getV() {
            return v;
        }

        public void setV(int v) {
            this.v = v;
        }

        public int getW() {
            return w;
        }

        public void setW(int w) {
            this.w = w;
        }

        public Good(int v, int w) {
            this.v = v;
            this.w = w;
        }
    }
}

