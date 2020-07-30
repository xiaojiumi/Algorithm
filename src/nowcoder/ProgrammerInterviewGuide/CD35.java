package nowcoder.ProgrammerInterviewGuide;
/**
 *题目描述
 * 给定一个数组arr，返回不包含本位置值的累乘数组
 * 例如，arr=[2,3,1,4]，返回[12, 8, 24, 6]，即除自己外，其他位置上的累乘
 * [要求]
 * 时间复杂度为O(n)O(n)，额外空间复杂度为O(1)O(1)
 *
 * 输入描述:
 * 第一行有两个整数N, P。分别表示序列长度，模数(即输出的每个数需要对此取模)
 * 接下来一行N个整数表示数组内的数
 * 输出描述:
 * 输出N个整数表示答案
 * 示例1
 * 输入
 * 4 100000007
 * 2 3 1 4
 * 输出
 * 12 8 24 6
 */
import java.util.Arrays;
import java.util.Scanner;


public class CD35 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int k= scanner.nextInt();
        int[] nums=new int[n];
        for (int i=0;i<n;i++){
            nums[i]= scanner.nextInt();
        }
        long[] left=new long[n];
        long[] right=new long[n];
        Arrays.fill(left,1);
        Arrays.fill(right,1);
        for(int i=1;i<n;i++){
            left[i]=left[i-1]*nums[i-1]%k;
        }
        for(int i=n-2;i>=0;i--){
            right[i]=right[i+1]*nums[i+1]%k;
        }
        for (int i=0;i<n;i++){
            System.out.print(left[i]*right[i]%k+" ");
        }
    }
}
