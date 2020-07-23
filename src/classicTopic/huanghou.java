package classicTopic;

public class huanghou {
    private static long upperlim = 1;
    private static int sum = 0;

    public static void main(String[] args) {
        upperlim = (upperlim << 8) - 1;
        huang(0, 0, 0);
        System.out.println(sum);
    }

    public static void huang(long row, long l, long r) {
        if (upperlim != row) {
            long pos = upperlim & ~(row | l | r);
            while (pos != 0) {
                long p = pos & -pos;
                pos -= p;
                huang(row + p, (l + p) << 1, (r + p) >> 1);
            }
        } else {
            sum++;
        }
    }
}

