package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定一个有序数组arr，调整arr使得这个数组的左半部分[1, \frac{n+1}{2}][1,
 * 2
 * n+1
 * ​
 *  ]没有重复元素且升序，而不用保证右部分是否有序
 * 例如，arr = [1, 2, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 8, 9]，调整之后arr=[1, 2, 3, 4, 5, 6, 7, 8, 9, .....]。
 * [要求]
 * 时间复杂度为O(n)O(n)，空间复杂度为O(1)O(1)
 *
 * 输入描述:
 * 第一行一个整数N。表示数组长度。
 * 接下来一行N个整数，表示数组内元素
 * 输出描述:
 * 输出N个整数为答案数组
 * 示例1
 * 输入
 * 16
 * 1 2 2 2 3 3 4 5 6 6 7 7 8 8 8 9
 * 输出
 * 1 2 3 4 5 6 7 8 9 6 2 7 2 8 8 3
 */

import java.util.Scanner;

public class CD36 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int[] nums=new int[n];
        for (int i=0;i<n;i++){
            nums[i]= scanner.nextInt();
        }
        int p=0;
        for (int i=1;i<n;i++){
            if (nums[i]!=nums[p]){
                p++;
                int temp=nums[i];
                nums[i]=nums[p];
                nums[p]=temp;
            }
        }
        for (int j=0;j<n;j++) System.out.print(nums[j]+" ");
    }
}
