package nowcoder.ProgrammerInterviewGuide;
/**题目描述
 给定一个整形数组arr，返回排序后相邻两数的最大差值
 arr = [9, 3, 1, 10]。如果排序，结果为[1, 3, 9, 10]，9和3的差为最大差值，故返回6。
 arr = [5, 5, 5, 5]。返回0。
 [要求]
 时间复杂度为O(n)O(n)，空间复杂度为O(n)O(n)

 输入描述:
 第一行一个整数N。表示数组长度。
 接下来N个整数表示数组内的元素
 输出描述:
 输出一个整数表示答案
 示例1
 输入
 4
 9 3 1 10
 输出
 6
 *
 */

import java.util.Arrays;
import java.util.Scanner;

public class CD40 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int[] nums=new int[n];
        for (int i=0;i<n;i++){
            nums[i]= scanner.nextInt();
        }
        Arrays.sort(nums);
        int max=0;
        for (int i=1;i<n;i++){
            max=Math.max(nums[i]-nums[i-1],max);
        }
        System.out.println(max);
    }
}
