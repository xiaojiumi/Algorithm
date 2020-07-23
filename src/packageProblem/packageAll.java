package packageProblem;

import java.util.PriorityQueue;
import java.util.Scanner;

public class packageAll {
    /**
     * f[i]体积j时最大总价值
     * 每件物品可以用无限次
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
            for (int j = v; j <= m; j++) {
                f[j] = Math.max(f[j], f[j - v] + w);
            }
        }
        System.out.println(f[m]);
    }

    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> q = new PriorityQueue<>();
        for (int num : nums) {
            q.add(num);
            if (q.size() > k) {
                q.poll();
            }
        }
        return q.peek();
    }
}
