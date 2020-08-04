package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 定义回文数的概念如下：
 * 如果一个非负数左右完全对应，则该数是回文数，例如：121,22等。
 * 如果一个负数的绝对值左右完全对应，也是回文数，例如：-121,-22等。
 * 给定一个32位整数num，判断num是否是回文数。
 * [要求]
 * O(log_{10} n)O(log
 * 10
 * ​
 *  n)
 * 输入描述:
 * 输入一个整数N.
 * 输出描述:
 * 若N是回文整数输出"Yes"，否则输出"No"
 * 示例1
 * 输入
 * 121
 * 输出
 * Yes
 */

import org.junit.Test;

import java.util.Scanner;

public class CD73 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String s= scanner.nextLine();
        if (s.charAt(0)=='-'){
            s=s.substring(1);
        }
        int i=0,j=s.length()-1;
        boolean b=true;
        while (i<j){
            if (s.charAt(i)!=s.charAt(j)){
                b=false;
                break;
            }
            i++;j--;
        }
        System.out.println(b?"Yes":"No");
    }
}

