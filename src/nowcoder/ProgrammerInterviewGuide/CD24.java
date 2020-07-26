package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定一个长度不小于2的数组arr，实现一个函数调整arr，要么让所有的偶数下标都是偶数，要么让所有的奇数下标都是奇数
 * 注意：1、数组下标从0开始！
 * 2、本题有special judge，你可以输出任意一组合法解！同时可以证明解一定存在
 * [要求]
 * 时间复杂度为O(n)O(n)，额外空间复杂度为O(1)O(1)
 *
 * 输入描述:
 * 第一行一个整数N。表示数组长度
 * 接下来一行N个整数表示数组内的数
 * 输出描述:
 * 输出N个整数。表示调整后的数组
 * 示例1
 * 输入
 * 5
 * 1 2 3 4 5
 * 输出
 * 2 1 4 3 5
 * 说明
 * 样例中的输出保证了奇数下标都是奇数
 */

import java.util.Scanner;

public class CD24 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int[] nums=new int[n];
        for (int i=0;i<n;i++){
            nums[i]= scanner.nextInt();
        }
        int odd=1,even=0;
        while (true){
            while (odd<n){
                if (nums[odd]%2==1)odd+=2;
                else break;
            }
            while (even<n){
                if (nums[even]%2==0)even+=2;
                else break;
            }
            if (odd>=n||even>=n)break;
            swap(odd,even,nums);
        }
        for (int i=0;i<n;i++) System.out.print(nums[i]+" ");
    }

    public static void swap(int i, int j, int[] nums){
        int temp=nums[i];
        nums[i]=nums[j];
        nums[j]=temp;
    }
}
