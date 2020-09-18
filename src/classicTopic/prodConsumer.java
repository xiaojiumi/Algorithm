package classicTopic;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public class prodConsumer {

    public static void main(String[] args) {
        shareData shareData=new shareData();
        new Thread(()->{
            for (int i = 0; i < 5; i++) {
                try {
                    shareData.incr();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        },"A").start();
        new Thread(()->{
            for (int i = 0; i < 5; i++) {
                try {
                    shareData.decr();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        },"B").start();
    }
}

class shareData{
    ReentrantLock lock=new ReentrantLock();
    int count=0;
    Condition condition=lock.newCondition();

    public void incr()throws Exception{
        lock.lock();
        try {
            while (count!=0){
                condition.await();
            }
            count++;
            System.out.println(Thread.currentThread().getName()+" "+count);
            condition.signalAll();
        }finally {
            lock.unlock();
        }
    }

    public void decr()throws Exception{
        lock.lock();
        try {
            while (count==0){
                condition.await();
            }
            count--;
            System.out.println(Thread.currentThread().getName()+" "+count);
            condition.signalAll();
        }finally {
            lock.unlock();
        }
    }
}
