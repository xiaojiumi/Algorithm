package nowcoder.ProgrammerInterviewGuide;

/**
 *题目描述
 * 给定排序数组arr和整数k，不重复打印arr中所有相加和为k的不降序二元组
 * 例如, arr = [-8, -4, -3, 0, 1, 2, 4, 5, 8, 9], k = 10，打印结果为：
 * 1, 9
 * 2, 8
 * [要求]
 * 时间复杂度为O(n)O(n)，空间复杂度为O(1)O(1)
 * 输入描述:
 * 第一行有两个整数n, k
 * 接下来一行有n个整数表示数组内的元素
 * 输出描述:
 * 输出若干行，每行两个整数表示答案
 * 按二元组从小到大的顺序输出(二元组大小比较方式为每个依次比较二元组内每个数)
 * 示例1
 * 输入
 * 10 10
 * -8 -4 -3 0 1 2 4 5 8 9
 * 输出
 * 1 9
 * 2 8
 */
import java.util.*;
public class CD3 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n= scanner.nextInt();
        int k= scanner.nextInt();
        int[] nums=new int[n];
        for (int i=0;i<n;i++)nums[i]=scanner.nextInt();
        int left=0,right=nums.length-1;
        while (left<right){
            if ((nums[left]+nums[right])<k){
                left++;
            }else if ((nums[left]+nums[right])>k)right--;
            else {
                if (left==0||nums[left]!=nums[left-1]||right==n-1){
                    System.out.println(nums[left]+" "+nums[right]);
                }
                left++;
                right--;
            }
        }
    }
}
