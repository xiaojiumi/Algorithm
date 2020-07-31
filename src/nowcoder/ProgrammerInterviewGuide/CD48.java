package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定两个升序链表，打印两个升序链表的公共部分。
 * 输入描述:
 * 第一个链表的长度为 n。
 *
 * 第二个链表的长度为 m。
 *
 * 链表结点的值为 val。
 * 输出描述:
 * 输出一行整数表示两个升序链表的公共部分的值 (按升序输出)。
 * 示例1
 * 输入
 * 4
 * 1 2 3 4
 * 5
 * 1 2 3 5 6
 * 输出
 * 1 2 3
 */

import java.util.ArrayList;
import java.util.Scanner;

public class CD48 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n1= scanner.nextInt();
        int[] nums1=new int[n1];
        for (int i=0;i<n1;i++){
            nums1[i]= scanner.nextInt();
        }
        int n2= scanner.nextInt();
        int[] nums2=new int[n2];
        for (int i=0;i<n2;i++){
            nums2[i]= scanner.nextInt();
        }
        int i=0,j=0;
        ArrayList<Integer> list=new ArrayList<>();
        while (i<n1&&j<n2){
            while (i<n1&&j<n2&&nums1[i]<nums2[j])i++;
            while (i<n1&&j<n2&&nums1[i]>nums2[j])j++;
            if (i<n1&&j<n2){
                if (nums1[i]==nums2[j]){
                    list.add(nums1[i]);
                    i++;
                    j++;
                }
            }

        }
        for (int a:list) System.out.print(a+" ");
    }
}
