package nowcoder.ProgrammerInterviewGuide;
/**
 *题目描述
 * 有N个长度不一的数组，所有的数组都是有序的，请从大到小打印这N个数组整体最大的前K个数。
 * 例如，输入含有N行元素的二维数组可以代表N个一维数组。
 * 219, 405, 538, 845, 971
 * 148, 558
 * 52, 99, 348, 691
 * 再输入整数k=5，则打印：
 * Top 5: 971, 845, 691, 558, 538
 * [要求]
 * 时间复杂度为O(k \log k)O(klogk)，空间复杂度为O(k \log k)O(klogk)
 *
 * 输入描述:
 * 第一行两个整数T, K。分别表示数组个数，需要打印前K大的元素
 * 接下来T行，每行输入格式如下：
 * 开头一个整数N，表示该数组的大小，接下来N个已排好序的数表示数组内的数
 * 输出描述:
 * 从大到小输出输出K个整数，表示前K大。
 * 示例1
 * 输入
 * 3 5
 * 5 219 405 538 845 971
 * 2 148 558
 * 4 52 99 348 691
 * 输出
 * 971 845 691 558 538
 */
import java.util.*;

public class CD34 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n=scanner.nextInt();
        int k= scanner.nextInt();
        PriorityQueue<Integer> queue=new PriorityQueue<>((o1, o2) -> o1-o2);
        while (scanner.hasNextInt()){
            int x=scanner.nextInt();
            queue.add(x);
            if (queue.size()>k) queue.poll();
        }
        ArrayList<Integer> list=new ArrayList<>();
        for (int x:queue)list.add(x);
        Collections.sort(list,(o1, o2) -> o2-o1);
        for (int x:list) System.out.print(x+" ");
    }
}
