package classicTopic;

import java.util.HashMap;

public class Fibonacci {

    public static void main(String[] args) {
        System.out.println(f3(50000));
    }

    public static int f1(int n){
        if (n==1||n==2)return 1;
        return f1(n-1)+f1(n-2);
    }

    static HashMap<Integer,Integer> map=new HashMap<>();
    public static int f2(int n){
        if (n==1||n==2)return 1;
        if (!map.containsKey(n)){
            map.put(n,f2(n-1)+f2(n-2));
        }
        return map.get(n);
    }

    public static long f3(int n){
        if (n==1||n==2)return 1;
        long a=1,b=1,c=0;
        for (int i=3;i<=n;i++){
            c=a+b;
            a=b;
            b=c;
        }
        return c;
    }
}
