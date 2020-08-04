package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定一个非负整数N，返回N!结果的末尾为0的数量
 *
 * 输入描述:
 * 第一行一个整数N。
 * 输出描述:
 * 输出一个整数表示N!的末尾0的数量。
 * 示例1
 * 输入
 * 3
 * 输出
 * 0
 * 说明
 * 3! = 6
 */

import java.util.Scanner;

public class CD56 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int ans=0;
        while (n>4){
            ans+=n/5;
            n/=5;
        }
        System.out.println(ans);
    }
}
