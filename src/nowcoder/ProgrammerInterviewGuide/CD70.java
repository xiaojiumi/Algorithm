package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定一个正数数组\ arr arr 和一个正数\ range range，可以选择\ arr arr 中的任意个数字加起来的和为\ sum sum。
 * 返回最小需要往\ arr arr 中添加几个数，使得\ sum sum 可以取到1 \sim range1∼range范围上的每一个数。
 * 给出的数组不保证有序！
 * 输入描述:
 * 第一行一个整数N, K。表示数组长度以及range
 * 接下来一行N个整数表示数组内的元素
 * 输出描述:
 * 输出一个整数表示答案
 * 示例1
 * 输入
 * 4 15
 * 1 2 3 7
 * 输出
 * 1
 * 说明
 * 想累加得到1 \sim 151∼15范围上的所有的数，arr还缺14这个数，所以返回1
 */

import java.util.PriorityQueue;
import java.util.Scanner;

public class CD70 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int target= scanner.nextInt();
        PriorityQueue<Integer> queue=new PriorityQueue<>((o1, o2) -> o1-o2);
        for (int i=0;i<n;i++){
            queue.add(scanner.nextInt());
        }
        int cur=0,res=0;
        while (cur<target){
            if (!queue.isEmpty()){
                int x=queue.peek();
                if (x<=cur+1){
                    cur+=x;
                    queue.poll();
                }else {
                    res++;
                    cur+=(cur+1);
                }
            }else {
                res++;
                cur+=(cur+1);
            }
        }
        System.out.println(res);
    }
}
