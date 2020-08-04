package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * N个加油站组成一个环形，给定两个长度都是N的非负数组oil和dis(N>1)，oil[i]代表第i个加油站存的油可以跑多少千米，dis[i]代表第i个加油站到环中下一个加油站相隔多少千米。假设你有一辆油箱足够大的车，初始时车里没有油。如果车从第i个加油站出发，最终可以回到这个加油站，那么第i个加油站就算良好出发点，否则就不算。请返回长度为N的boolean型数组res，res[i]代表第i个加油站是不是良好出发点
 * 规定只能按照顺时针走，也就是i只能走到i+1，N只能走到1
 * [要求]
 * 时间复杂度为O(n)O(n)，空间复杂度为O(1)O(1)
 * 输入描述:
 * 第一行一个整数N表示加油站数量。
 * 第二行N个整数，表示oil数组。
 * 第三行N个整数，表示dis数组。
 * 输出描述:
 * 输出N个整数。若第i个整数为0表示该位置不是良好出发点，为1表示该位置是良好出发点。
 * 示例1
 * 输入
 * 复制
 * 9
 * 4 2 0 4 5 2 3 6 2
 * 6 1 3 1 6 4 1 1 6
 * 输出
 * 复制
 * 0 0 0 0 0 0 0 0 0
 * 示例2
 * 输入
 * 8
 * 4 5 3 1 5 1 1 9
 * 1 9 1 2 6 0 2 0
 * 输出
 * 0 0 1 0 0 1 0 1
 * 说明
 * 如果车从A点出发，到B点且加上B的油，还剩8的油，发现到不了C；
 * 如果从B点出发，发现车到不了C；
 * 如果从C点出发，发现可以转一圈，所以C点是良好出发点。
 * \dots \dots……
 */

import java.util.Arrays;
import java.util.Scanner;

public class CD53 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n= scanner.nextInt();
        long[] oils=new long[n];
        long[] dis=new long[n];
        for (int i=0;i<n;i++){
            oils[i]= scanner.nextLong();
        }
        for (int i=0;i<n;i++){
            dis[i]= scanner.nextLong();
        }
        int[] ans=new int[n];
        Arrays.fill(ans,-1);
        for (int i=0;i<n;i++)oils[i]-=dis[i];
        int start=0,end=1;
        long rest= oils[0];
        while (start!=end){
            if (rest>=0){
                rest+= oils[end];
                end=(end+1)%n;
            }else {
                start=(start-1+n)%n;
                rest+= oils[start];
            }
        }
        if (rest>=0){
            int cur=start;
            rest= oils[start];
            while (ans[cur]==-1){
                if (rest>=0){
                    ans[cur]=1;
                    cur=(cur-1+n)%n;
                    rest= oils[cur];
                }else {
                    ans[cur]=0;
                    cur=(cur-1+n)%n;
                    rest+= oils[cur];
                }
            }
        }else {
            int cur=start;
            while (ans[cur]==-1){
                ans[cur]=0;
                cur=(cur-1+n)%n;
            }
        }
        for (int i:ans) {
            System.out.print(i+" ");
        }
    }
}
