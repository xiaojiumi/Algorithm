package test;

import java.util.Comparator;
import java.util.TreeSet;

public class treemao {
    public static void main(String[] args) {
        TreeSet<man> set=new TreeSet<>();
        man m1 = new man(1, "asd");
        man m2 = new man(2, "asad");
        man m3 = new man(3, "asdsdd");
        set.add(m3);
        set.add(m1);
        set.add(m2);

        for (man m:set){
            System.out.println(m);
        }
    }
}
