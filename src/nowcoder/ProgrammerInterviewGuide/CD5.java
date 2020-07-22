package nowcoder.ProgrammerInterviewGuide;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;
import java.util.Stack;

/**
 * 题目描述
 * 实现一个特殊功能的栈，在实现栈的基本功能的基础上，再实现返回栈中最小元素的操作。
 * 输入描述:
 * 第一行输入一个整数N，表示对栈进行的操作总数。
 *
 * 下面N行每行输入一个字符串S，表示操作的种类。
 *
 * 如果S为"push"，则后面还有一个整数X表示向栈里压入整数X。
 *
 * 如果S为"pop"，则表示弹出栈顶操作。
 *
 * 如果S为"getMin"，则表示询问当前栈中的最小元素是多少。
 * 输出描述:
 * 对于每个getMin操作，输出一行表示当前栈中的最小元素是多少。
 * 示例1
 * 输入
 * 6
 * push 3
 * push 2
 * push 1
 * getMin
 * pop
 * getMin
 * 输出
 * 1
 * 2
 */
import java.util.Scanner;
import java.util.Stack;
public class CD5 {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n= scanner.nextInt();
        Stack<Integer> s1=new Stack<>();
        Stack<Integer> s2=new Stack<>();
        for (int i=0;i<n+1;i++){
            String str= scanner.nextLine();
            if (str.contains("push")){
                String[] string = str.split(" ");
                Integer cur = Integer.valueOf(string[1]);
                s1.add(cur);
                if (s2.isEmpty()){
                    s2.add(cur);
                }else {
                    if (cur<s2.peek()){
                        s2.add(cur);
                    }else s2.add(s2.peek());
                }
            }else if (str.contains("get")){
                System.out.println(s2.peek());
            }else if (str.contains("pop")){
                s1.pop();
                s2.pop();
            }
        }
    }
}
