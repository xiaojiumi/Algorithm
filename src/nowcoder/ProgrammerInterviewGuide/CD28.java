package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 定义局部最小的概念。arr长度为1时，arr[0]是局部最小。arr的长度为N(N>1)时，如果arr[0]<arr[1]，那么arr[0]是局部最小；如果arr[N-1]<arr[N-2]，那么arr[N-1]是局部最小；如果0<i<N-1，既有arr[i]<arr[i-1]，又有arr[i]<arr[i+1]，那么arr[i]是局部最小。
 * 给定无序数组arr，已知arr中任意两个相邻的数不相等。写一个函数，只需返回arr中任意一个局部最小出现的位置即可
 * [要求]
 * 时间复杂度为O(\log n)O(logn)，空间复杂度为O(1)O(1)
 *
 * 输入描述:
 * 第一行有一个整数N。标书数组长度
 * 接下来一行，每行N个整数表示数组中的数
 * 输出描述:
 * 输出一个整数表示答案
 * 示例1
 * 输入
 * 3
 * 2 1 3
 * 输出
 * 1
 */

import java.util.Scanner;

public class CD28 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int[] nums=new int[n];
        for (int i=0;i<n;i++){
            nums[i]= scanner.nextInt();
        }
        if (n==1) System.out.println(1);
        if (nums[0]<nums[1]) System.out.println(0);
        if (nums[n-2]>nums[n-1]) System.out.println(n-1);
        int left=1,right=n-2;
        while (left<right){
            int mid=(left+right)/2;
            if (nums[mid]>nums[mid-1]){
                right=mid-1;
            }else if (nums[mid]>nums[mid+1]){
                left=mid+1;
            }else {
                System.out.println(mid);
                return;
            }
        }
        System.out.println(left);
    }
}
