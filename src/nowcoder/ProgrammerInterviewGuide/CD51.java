package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定一个正数数组arr，arr的累加和代表金条的总长度，arr的每个数代表金条要分成的长度。规定长度为k的金条分成两块，费用为k个铜板。返回把金条分出arr中的每个数字需要的最小代价。
 * [要求]
 * 时间复杂度为O(n \log n)O(nlogn)，空间复杂度为O(n)O(n)
 *
 * 输入描述:
 * 第一行一个整数N。表示数组长度。
 *
 * 接下来一行N个整数，表示arr数组。
 * 输出描述:
 * 一个整数表示最小代价
 * 示例1
 * 输入
 * 3
 * 10 30 20
 * 输出
 * 90
 * 说明
 * 如果先分成40和20两块，将花费60个铜板，再把长度为40的金条分成10和30两块，将花费40个铜板，总花费为100个铜板；
 *
 * 如果先分成10和50两块，将花费60个铜板，再把长度为50的金条分成20和30两块，将花费50个铜板，总花费为110个铜板；
 *
 * 如果先分成30和30两块，将花费60个铜板，再把其中一根长度为30的金条分成10和20两块，将花费30个铜板，总花费为90个铜板；
 *
 * 因此最低花费为90
 */

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Scanner;

public class CD51 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n= scanner.nextInt();
        long[] nums=new long[n];
        for (int i=0;i<n;i++){
            nums[i]= scanner.nextLong();
        }
        PriorityQueue<Long> queue=new PriorityQueue<>();
        for (long num:nums)queue.offer(num);
        long res=0;
        while (queue.size()>1){
            long temp=queue.poll()+queue.poll();
            res+=temp;
            queue.offer(temp);
        }
        System.out.println(res);
    }
}
