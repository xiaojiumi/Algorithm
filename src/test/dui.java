package test;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;

public class dui {

    public static void main(String[] args) {
        BlockingQueue<Integer> queue=new ArrayBlockingQueue<>(5);
        queue.offer(1);
        queue.offer(7);
        queue.offer(3);
        queue.offer(5);
        queue.offer(6);
        queue.offer(7);
        queue.offer(8);
        queue.offer(9);
        queue.offer(2);

        System.out.println(queue.poll());
    }
}
