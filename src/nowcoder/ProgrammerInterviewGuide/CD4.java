package nowcoder.ProgrammerInterviewGuide;



/**
 * 题目描述
 * 给定排序数组arr和整数k，不重复打印arr中所有相加和为k的严格升序的三元组
 * 例如, arr = [-8, -4, -3, 0, 1, 2, 4, 5, 8, 9], k = 10，打印结果为：
 * -4 5 9
 * -3 4 9
 * -3 5 8
 * 0 1 9
 * 0 2 8
 * 1 4 5
 * [要求]
 * 时间复杂度为O(n^2)O(n
 * 2
 *  )，空间复杂度为O(1)O(1)
 *
 * 输入描述:
 * 第一行有两个整数n, k
 * 接下来一行有n个整数表示数组内的元素
 * 输出描述:
 * 输出若干行，每行三个整数表示答案
 * 按三元组从小到大的顺序输出(三元组大小比较方式为每个依次比较三元组内每个数)
 * 示例1
 * 输入
 * 10 10
 * -8 -4 -3 0 1 2 4 5 8 9
 * 输出
 * -4 5 9
 * -3 4 9
 * -3 5 8
 * 0 1 9
 * 0 2 8
 * 1 4 5
 */
import java.util.Scanner;
public class CD4 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n= scanner.nextInt();
        int k= scanner.nextInt();
        int[] nums=new int[n];
        for (int i=0;i<n;i++)nums[i]=scanner.nextInt();
        for (int a=0;a<n-2;a++){
            if (a>0&&nums[a]==nums[a-1])continue;
            int b=a+1,c=n-1;
            while (b<c){
                if (nums[a]+nums[b]+nums[c]<k){
                    b++;
                }else if (nums[a]+nums[b]+nums[c]>k){
                    c--;
                }else {
                    if (nums[b]!=nums[b-1]){
                        System.out.println(nums[a]+" "+nums[b]+" "+nums[c]);
                    }
                    b++;
                    c--;
                }
            }
        }
    }
}
