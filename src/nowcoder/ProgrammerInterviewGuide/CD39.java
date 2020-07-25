package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给定一个无序数组arr，找到数组中未出现的最小正整数
 * 例如arr = [-1, 2, 3, 4]。返回1
 *        arr = [1, 2, 3, 4]。返回5
 * [要求]
 * 时间复杂度为O(n)O(n)，空间复杂度为O(1)O(1)
 *
 * 输入描述:
 * 第一行为一个整数N。表示数组长度。
 * 接下来一行N个整数表示数组内的数
 * 输出描述:
 * 输出一个整数表示答案
 * 示例1
 * 输入
 * 4
 * -1 2 3 4
 * 输出
 * 1
 */

import java.util.HashMap;
import java.util.Scanner;

public class CD39 {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        HashMap<Integer,Integer> map=new HashMap<>();
        for(int i=0;i<n;i++){
                int x=scanner.nextInt();
                map.put(x,1);
        }
        for (int i=1;i<=n;i++){
            if (!map.containsKey(i)){
                System.out.println(i);
                return;
            }

        }
        System.out.println(n+1);
    }
}
