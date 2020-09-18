package classicTopic;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class blockingqueue {

    private AtomicInteger atomicInteger=new AtomicInteger();
    private volatile boolean flag=true;
    BlockingQueue<String> queue;

    public blockingqueue(BlockingQueue queue) {
        this.queue = queue;
    }

    public void stop(){
        this.flag=false;
    }

    public void prod()throws Exception{
        String data=null;
        boolean res;
        while (flag){
            data=atomicInteger.incrementAndGet()+" ";
            res = queue.offer(data, 1L, TimeUnit.SECONDS);
            if (res){
                System.out.println(Thread.currentThread().getName()+" "+atomicInteger);
            }
            Thread.sleep(10);
        }
        System.out.println("prod stop");
    }

    public void con()throws Exception{
        String data=null;
        while (flag){
            data = queue.poll(1L, TimeUnit.SECONDS);
            if (data==null||data.equalsIgnoreCase("")){
                flag=false;
                System.out.println("con stop");
                return;
            }
            System.out.println(Thread.currentThread().getName()+" "+data);
            Thread.sleep(1000);
        }
    }

    public static void main(String[] args) throws Exception{
        blockingqueue blockingqueue=new blockingqueue(new ArrayBlockingQueue(10));
        new Thread(()->{
            try {
                blockingqueue.prod();
            } catch (Exception e) {
                e.printStackTrace();
            }
        },"A").start();
        new Thread(()->{
            try {
                blockingqueue.con();
            } catch (Exception e) {
                e.printStackTrace();
            }
        },"B").start();
        try { TimeUnit.SECONDS.sleep(5); }catch (InterruptedException e){ e.printStackTrace(); }
        blockingqueue.stop();
    }
}
