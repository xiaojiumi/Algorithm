package packageProblem;

import java.util.ArrayList;
import java.util.Scanner;

public class packageMultipleOpti {
    /**
     * f[i]体积j时最大总价值
     * 每件物品可以用si次
     * 物品总数很多，三重循环时间复杂度很高
     * 优化：将多重背包转化成01背包
     * 若只是将1000个同类物品当成1000种物品，时间复杂度不变
     * 表示方法：7->1+2+4表示 10->1+2+4+3表示
     * 将物品用二进制优化
     * 时间复杂度为logV,大约10^7,c语言1秒能运算10^7~10^8
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
}

class Good {
    int v;
    int w;

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
