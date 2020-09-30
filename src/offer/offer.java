package offer;

import java.util.HashSet;

public class offer {

    public static void main(String[] args) {

    }



    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix==null||matrix.length==0)return false;
        int n= matrix.length,m= matrix[0].length;
        int i=0,j=m-1;
        while (i>=0&&i<n&&j>=0&&j<m){
            int temp= matrix[i][j];
            if (target>temp){
                i++;
            }else if (target<temp){
                j--;
            }else return true;
        }
        return false;
    }

    public int findRepeatNumber(int[] nums) {
        HashSet<Integer> set=new HashSet<>();
        for (int n:nums){
            if (!set.contains(n)){
                set.add(n);
            }else {
                return n;
            }
        }
        return -1;
    }



}
