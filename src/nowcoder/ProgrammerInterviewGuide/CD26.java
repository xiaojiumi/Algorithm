package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定一个数组arr，返回子数组的最大累加和
 * 例如，arr = [1, -2, 3, 5, -2, 6, -1]，所有子数组中，[3, 5, -2, 6]可以累加出最大的和12，所以返回12.
 * [要求]
 * 时间复杂度为O(n)O(n)，空间复杂度为O(1)O(1)
 *
 * 输入描述:
 * 第一行一个整数N。表示数组长度
 * 接下来一行N个整数表示数组内的元素
 * 输出描述:
 * 输出一个整数表示答案
 * 示例1
 * 输入
 * 7
 * 1 -2 3 5 -2 6 -1
 * 输出
 * 12
 */

import java.util.Scanner;

public class CD26 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int[] nums=new int[n];
        for (int i=0;i<n;i++){
            nums[i]= scanner.nextInt();
        }
        int ans=nums[0];
        for (int i=1;i<n;i++){
            if (nums[i-1]>0)nums[i]+=nums[i-1];
            ans=Math.max(ans,nums[i]);
        }
        System.out.println(ans);
    }
}
