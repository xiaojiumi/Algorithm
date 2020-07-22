package nowcoder.ProgrammerInterviewGuide;

import java.util.Scanner;
import java.util.Stack;

/**
 * 题目描述
 * 用两个栈实现队列，支持队列的基本操作。
 * 输入描述:
 * 第一行输入一个整数N，表示对队列进行的操作总数。
 *
 * 下面N行每行输入一个字符串S，表示操作的种类。
 *
 * 如果S为"add"，则后面还有一个整数X表示向队列尾部加入整数X。
 *
 * 如果S为"poll"，则表示弹出队列头部操作。
 *
 * 如果S为"peek"，则表示询问当前队列中头部元素是多少。
 * 输出描述:
 * 对于每一个为"peek"的操作，输出一行表示当前队列中头部元素是多少。
 * 示例1
 * 输入
 * 6
 * add 1
 * add 2
 * add 3
 * peek
 * poll
 * peek
 * 输出
 * 1
 * 2
 */
import java.util.Scanner;
import java.util.Stack;
public class CD6 {

    static Stack<Integer> a = new Stack<>();
    static Stack<Integer> b = new Stack<>();
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String line1 = scanner.nextLine();
        int n = Integer.valueOf(line1);
        for (int i = 0; i < n; i++) {
            String str = scanner.nextLine();
            if (str.contains("add")) {
                Integer cur = Integer.valueOf(str.split(" ")[1]);
                add(cur);
            } else if (str.contains("peek")) {
                System.out.println(peek());
            } else if (str.contains("poll")) {
               poll();
            }
        }
    }

    public static void add(int num){
        a.add(num);
        if (b.isEmpty()){
            while (!a.isEmpty()){
                b.add(a.pop());
            }
        }
    }

    public static void poll(){
        if (b.isEmpty()&&!a.isEmpty()){
            while (!a.isEmpty()){
                b.add(a.pop());
            }
        }
        b.pop();
    }

    public static int peek(){
        if (b.isEmpty()&&!a.isEmpty()){
            while (!a.isEmpty()){
                b.add(a.pop());
            }
        }
        return b.peek();
    }
}
