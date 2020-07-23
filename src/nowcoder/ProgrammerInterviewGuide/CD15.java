package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 有一个整型数组arr和一个大小为w的窗口从数组的最左边滑到最右边，窗口每次向右边滑一个位置，求每一种窗口状态下的最大值。（如果数组长度为n，窗口大小为w，则一共产生n-w+1个窗口的最大值）
 * 输入描述:
 * 第一行输入n和w，分别代表数组长度和窗口大小
 * 第二行输入n个整数X_iX
 * i
 * ​
 *  ，表示数组中的各个元素
 * 输出描述:
 * 输出一个长度为n-w+1的数组res，res[i]表示每一种窗口状态下的最大值
 * 示例1
 * 输入
 * 8 3
 * 4 3 5 4 3 3 6 7
 * 输出
 * 5 5 5 4 6 7
 */

import java.util.LinkedList;
import java.util.Scanner;

public class CD15 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int w=scanner.nextInt();
        int[] nums=new int[n];
        for (int i=0;i<n;i++){
            nums[i]= scanner.nextInt();
        }
        int[] ans=new int[n-w+1];
        int index=0;
        LinkedList<Integer> list=new LinkedList<>();
        for (int i=0;i<nums.length;i++){
            while (!list.isEmpty()&&nums[list.peekLast()]<=nums[i])list.pollLast();
            list.addLast(i);
            if (list.peekFirst()==i-w)list.pollFirst();
            if (i>=w-1)ans[index++]=nums[list.peekFirst()];
        }
        for (int i=0;i<ans.length;i++){
            System.out.print(ans[i]+" ");
        }
        }
    }

