package nowcoder.ProgrammerInterviewGuide;



/**
 * 题目描述
 * 给定一个数组arr，该数组无序，但每个值均为正数，再给定一个正数k。求arr的所有子数组中所有元素相加和为k的最长子数组的长度
 * 例如，arr = [1, 2, 1, 1, 1], k = 3
 * 累加和为3的最长子数组为[1, 1, 1]，所以结果返回3
 * [要求]
 * 时间复杂度为O(n)O(n)，空间复杂度为O(1)O(1)
 *
 * 输入描述:
 * 第一行两个整数N, k。N表示数组长度，k的定义已在题目描述中给出
 * 第二行N个整数表示数组内的数
 * 输出描述:
 * 输出一个整数表示答案
 * 示例1
 * 输入
 * 5 3
 * 1 2 1 1 1
 * 输出
 * 3
 */
import java.util.Scanner;
public class CD8 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int k=scanner.nextInt();
        int[] nums=new int[n];
        for (int i=0;i<n;i++){
            nums[i]=scanner.nextInt();
        }
        int start=0,end=0,sum=0,max=Integer.MIN_VALUE;
        while (start<n&&end<n){
            while (end<n){
                sum+=nums[end];
                end++;
                if (sum==k){
                    max=Math.max(max,end-start);
                }else if (sum>k)break;
            }
            while (start<n){
                sum-=nums[start];
                start++;
                if (sum==k){
                    max=Math.max(max,end-start);
                }else if (sum<k)break;
            }
        }
        System.out.println(max);
    }
}
