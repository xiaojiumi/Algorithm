package test;

public class man implements Comparable<man>{
    public int id;
    public String name;

    public man(int id, String name) {
        this.id = id;
        this.name = name;
    }

    @Override
    public String toString() {
        return "man{" +
                "id=" + id +
                ", name='" + name + '\'' +
                '}';
    }


    @Override
    public int compareTo(man o) {
        return this.name.length()-o.name.length();
    }
}
