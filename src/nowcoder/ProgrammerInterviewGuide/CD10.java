package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定一个无序数组arr，其中元素可正、可负、可0。求arr所有子数组中正数与负数个数相等的最长子数组的长度。
 * [要求]
 * 时间复杂度为O(n)O(n)，空间复杂度为O(n)O(n)
 * 输入描述:
 * 第一行一个整数N，表示数组长度
 * 接下来一行有N个数表示数组中的数
 * 输出描述:
 * 输出一个整数表示答案
 * 示例1
 * 输入
 * 5
 * 1 -2 1 1 1
 * 输出
 * 2
 */

import java.util.HashMap;
import java.util.Scanner;
public class CD10 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int[] nums=new int[n];
        for (int i=0;i<n;i++){
            nums[i]=scanner.nextInt();
        }
        HashMap<Integer, Integer> map=new HashMap<>();
        map.put(0,-1);
        int max=0,count=0;
        for (int i=0;i<n;i++)
        {
            count+=nums[i]==0?0:(nums[i]/Math.abs(nums[i]));
            if (map.containsKey(count)){
                max=Math.max(max,i-map.get(count));
            }else map.put(count,i);
        }
        System.out.println(max);
    }
}
