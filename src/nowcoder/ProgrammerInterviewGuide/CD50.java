package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定两个整数W和K，W代表你拥有的初始资金，K代表你最多可以做K个项目。再给定两个长度为N的正数数组costs[]和profits[]，代表一共有N个项目，costs[i]和profits[i]分别表示第i号项目的启动资金与做完后的利润(注意是利润，如果一个项目的启动资金为10，利润为4，代表该项目最终的收入为14)。你不能并行只能串行地做项目，并且手里拥有的资金大于或等于某个项目的启动资金时，你才能做这个项目。该如何选择做项目，能让你最终的收益最大？返回最后能获得的最大资金
 * [要求]
 * 时间复杂度为O(k \log n)O(klogn)，空间复杂度为O(n)O(n)
 * 输入描述:
 * 第一行三个整数N, W, K。表示总的项目数量，初始资金，最多可以做的项目数量
 * 第二行有N个正整数，表示costs数组
 * 第三行有N个正整数，表示profits数组
 * 输出描述:
 * 输出一个整数，表示能获得的最大资金
 * 示例1
 * 输入
 * 4 3 2
 * 5 4 1 2
 * 3 5 3 2
 * 输出
 * 11
 * 说明
 * 初始资金为3，最多做两个项目，每个项目的启动资金与利润见costs和profits。最优选择为：先做2号项目，做完之后资金增长到6,。然后做1号项目，做完之后资金增长到11。其他的任何选择都不会比这种选择好，所以返回11
 */

import java.util.PriorityQueue;
import java.util.Scanner;

public class CD50 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n= scanner.nextInt();
        int w= scanner.nextInt();
        int k= scanner.nextInt();
        int[] costs=new int[n];
        for (int i=0;i<n;i++){
            costs[i]= scanner.nextInt();
        }
        int[] profits=new int[n];
        for (int i=0;i<n;i++){
            profits[i]= scanner.nextInt();
        }
        long max=getMax(costs,profits,k,w);
        System.out.println(max);
    }

    public static long getMax(int[] costs,int[] profits,int k,int w){
        long max=w;
        PriorityQueue<Integer> minHeap=new PriorityQueue<>((o1,o2)->(costs[o1]-costs[o2]));
        PriorityQueue<Integer> maxHeap=new PriorityQueue<>((o1,o2)->(profits[o2]-profits[o1]));
        for (int i=0;i< costs.length;i++){
            minHeap.offer(i);
        }
        while (!minHeap.isEmpty()&&costs[minHeap.peek()]<=w){
            maxHeap.offer(minHeap.poll());
        }
        for (int i=0;i<k;i++){
            int profit=profits[maxHeap.poll()];
            max+=profit;
            w+=profit;
            while (!minHeap.isEmpty()&&costs[minHeap.peek()]<=w){
                maxHeap.offer(minHeap.poll());
            }
        }
        return max;
    }
}
