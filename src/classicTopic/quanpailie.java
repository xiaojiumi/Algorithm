package classicTopic;

import java.util.ArrayList;
import java.util.List;

public class quanpailie {
    List<List<Integer>> ans=new ArrayList<>();
    int[] nums;
    public List<List<Integer>> permute(int[] nums) {
        this.nums=nums;
        backtrack(new ArrayList<>(),new boolean[nums.length]);
        return ans;
    }

    public  void backtrack(List<Integer> cur,boolean[] b){
        if (cur.size()== nums.length){
            ans.add(new ArrayList<>(cur));
            return;
        }
        for (int i=0;i< nums.length;i++){
            if (b[i])continue;
            cur.add(nums[i] );
            b[i]=true;
            backtrack(cur,b);
            cur.remove(cur.size()-1);
            b[i]=false;
        }
    }

    public static void main(String[] args) {
        quanpailie q=new quanpailie();
        int[] a={1,2,3};
        System.out.println(q.permute(a));
    }
}
