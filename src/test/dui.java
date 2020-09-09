package test;

import java.util.Comparator;
import java.util.PriorityQueue;

public class dui {

    public static void main(String[] args) {
        PriorityQueue<Integer> queue=new PriorityQueue<>((o1, o2) -> o2-o1);
        queue.add(1);
        queue.add(7);
        queue.add(3);
        queue.add(5);
        for (int i:queue) System.out.println(i);
        System.out.println(queue.poll());
    }
}
