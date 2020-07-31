package nowcoder.ProgrammerInterviewGuide;
/**
 * 题目描述
 * 给出一个单链表，返回删除单链表的倒数第 K 个节点的链表。
 * 输入描述:
 * n 表示链表的长度。
 * val 表示链表中节点的值。
 * 输出描述:
 * 在给定的函数内返回链表的头指针。
 * 示例1
 * 输入
 * 5 4
 * 1 2 3 4 5
 * 输出
 * 1 3 4 5
 */

import java.util.Scanner;

public class CD49 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n= scanner.nextInt();
        int target= scanner.nextInt();
        Node prev=new Node(-1);
        Node cur=prev;
        for (int i=0;i<n;i++){
            Node node = new Node(scanner.nextInt());
            cur.next=node;
            cur=cur.next;
        }
        cur=prev;
        while (n-target>0){
            cur=cur.next;
            target++;
        }
        cur.next=cur.next.next;
        cur=prev.next;
        while (cur!=null){
            System.out.print(cur.value+" ");
            cur=cur.next;
        }
    }
}
class Node{
    int value;
    Node next;

    public Node(int value) {
        this.value = value;
    }


}