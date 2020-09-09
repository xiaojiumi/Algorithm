package mianshi;

import javax.sound.midi.SysexMessage;

public class prepare {

    public int maxSubArray(int[] nums) {
        int max=nums[0];
        for (int i=1;i< nums.length;i++){
            if (nums[i-1]>0)nums[i]+=nums[i-1];
            max=Math.max(max,nums[i]);
        }
        return max;
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int[] temp=new int[m];
        System.arraycopy(nums1,0,temp,0,m);
        int i=0,j=0,k=0;
        while (i<m&&j<n){
            if (temp[i]<nums2[j]){
                nums1[k++]=temp[i++];
            }else {
                nums1[k++]=nums2[j++];
            }
        }
        if (i<m) System.arraycopy(temp,i,nums1,k,m-i);
        if (j<n) System.arraycopy(nums2,j,nums1,k,m-j);
    }
}
