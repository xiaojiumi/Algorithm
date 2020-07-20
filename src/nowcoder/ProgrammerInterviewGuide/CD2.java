package nowcoder.ProgrammerInterviewGuide;

import java.util.*;

/**
 * 题目描述
 * 先给出可整合数组的定义：如果一个数组在排序之后，每相邻两个数的差的绝对值都为1，或者该数组长度为1，则该数组为可整合数组。例如，[5, 3, 4, 6, 2]排序后为[2, 3, 4, 5, 6]，符合每相邻两个数差的绝对值都为1，所以这个数组为可整合数组
 * 给定一个数组arr, 请返回其中最大可整合子数组的长度。例如，[5, 5, 3, 2, 6, 4, 3]的最大可整合子数组为[5, 3, 2, 6, 4]，所以请返回5
 * [要求]
 * 时间复杂度为O(n^2)O(n
 * 2
 *  )，空间复杂度为O(n)O(n)
 * 输入描述:
 * 第一行一个整数N，表示数组长度
 * 第二行N个整数，分别表示数组内的元素
 * 输出描述:
 * 输出一个整数，表示最大可整合子数组的长度
 *
 * 示例1
 * 输入
 * 7 5 5 3 2 6 4 3
 * 输出
 * 5
 */
public class CD2 {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int[] data=new int[n];
        for (int i=0;i<n;i++){
            data[i]=scanner.nextInt();
        }
        Arrays.sort(data);
        int[] dp=new int[n];
        Arrays.fill(dp,1);
        for (int i=1;i<n;i++){
            if (data[i]-data[i-1]==1){
                dp[i]=dp[i-1]+1;
            }else if (data[i]==data[i-1]){
                dp[i]=dp[i-1];
            }
        }
        OptionalInt max= Arrays.stream(dp).max();
        int asInt = max.getAsInt();
        System.out.println(asInt);
    }
}
