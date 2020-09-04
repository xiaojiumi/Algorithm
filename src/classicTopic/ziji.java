package classicTopic;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class ziji {
    List<List<Integer>> ans=new ArrayList<>();
    int[] nums;

    public List<List<Integer>> subsets(int[] nums) {
        this.nums=nums;
        for (int i=0;i<= nums.length;i++){
            backtrack(new ArrayList<>(),0,i);
        }
        return ans;
    }

    public void backtrack(List<Integer> cur,int start,int len){
        if (cur.size()==len){
            ans.add(new ArrayList<>(cur));
            return;
        }
        for (int i=start;i< nums.length;i++){
            cur.add(nums[i] );
            backtrack(cur,i+1,len);
            cur.remove(cur.size()-1);
        }
    }

    public static void main(String[] args) {
        ziji z=new ziji();
        int[] a={1,2,3};
        System.out.println(z.subsets(a));
    }
}
