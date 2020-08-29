package classicTopic;


public class kmp {

    public static void main(String[] args) {
        String s1="ABCDAABCDABD";
        String s2="ABCDABD";
        int[] next = getNext(s2);
        System.out.println(next);
        System.out.println(kmp(s1,s2));
    }

    public static int kmp(String query,String pattern){
        int n= query.length();
        int m=pattern.length();
        int[] next=getNext(pattern);
        int i=0,j=0;
        while (i<n&&j<m){
            if (j==-1||query.charAt(i)==pattern.charAt(j)){
                i++;
                j++;
            }else {
                j=next[j];
            }
        }
        if (j==m)return i-j;
        return -1;
    }

    // 已知next[j] = k,利用递归的思想求出next[j+1]的值
    // 如果已知next[j] = k,如何求出next[j+1]呢?具体算法如下:
    // 1. 如果p[j] = p[k], 则next[j+1] = next[k] + 1;
    // 2. 如果p[j] != p[k], 则令k=next[k],如果此时p[j]==p[k],则next[j+1]=k+1,
    // 如果不相等,则继续递归前缀索引,令 k=next[k],继续判断,直至k=-1(即k=next[0])或者p[j]=p[k]为止
    public static int[] getNext(String s){
        int len=s.length();
        int[] next=new int[len];
        int k=-1,j=0;
        next[0]=-1;
        while (j<len-1){
            if (k==-1||s.charAt(j)==s.charAt(k)){
                k++;
                j++;
                next[j]=k;
            }else {
                k= next[k];
            }
        }
        return next;
    }
}
