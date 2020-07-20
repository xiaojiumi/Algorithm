package leetcode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class topMianshi {

    public static void main(String[] args) {
    
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int n=nums1.length,m=nums2.length;
        int left=(n+m+1)/2,right=(n+m+2)/2;
        return (getK(nums1,0,n-1,nums2,0,m-1,left)+
                getK(nums1,0,n-1,nums2,0,m-1,right))*0.5;
    }

    public int getK(int[] nums1, int start1, int end1,
                    int[] nums2, int start2, int end2, int k){
        int len1=end1-start1+1,len2=end2-start2+1;
        if (len1>len2)return getK(nums2,start2,end2,nums1,start1,end1,k);
        if (len1==0)return nums2[start2+k-1];
        if (k==1)return Math.min(nums1[start1],nums2[start2]);
        int i=start1+ Math.min(len1,k/2)-1;
        int j=start2+ Math.min(len2,k/2)-1;
        if (nums1[i]>nums2[j]){
            return getK(nums1,start1,end1,nums2,j+1,end2,k-(j-start2+1));
        }else {
            return getK(nums1,i+1,end1,nums2,start2,end2,k-(i-start1+1));
        }
    }

    public String longestPalindrome(String s) {
        if (s==null||s.length()==0)return "";
        int start=0,end=0;
        for (int i=0;i<s.length();i++){
            int len1=expandFromCenter(s,i,i);
            int len2=expandFromCenter(s,i,i+1);
            int len= Math.max(len1,len2);
            if (len>end-start){
                start=i-(len-1)/2;
                end=i+len/2;
            }
        }
        return s.substring(start,end+1);
    }

    public int expandFromCenter(String s, int left, int right){
        int L=left,R=right;
        while (L>=0&&R<s.length()&&s.charAt(L)==s.charAt(R)){
            L--;
            R++;
        }
        return R-L-1;
    }

    public int searchInsert(int[] nums, int target) {
        int left=0,right=nums.length-1;
        while(left<=right){
            int mid=left+(right-left)/2;
            if (nums[mid]>target){
                right=mid-1;
            }else if (nums[mid]<target){
                left=mid+1;
            }else if (nums[mid]==target){
                return mid;
            }
        }
        return left;
    }

    public int reverse(int x) {
        int ans=0;
        while (x!=0){
            int temp=x%10;
            x=x/10;
            if (ans> Integer.MAX_VALUE/10||
                    (ans== Integer.MAX_VALUE/10&&temp> Integer.MAX_VALUE%10)){
                ans=0;
                break;
            }else if (ans< Integer.MIN_VALUE/10||
                    (ans== Integer.MIN_VALUE/10&&temp< Integer.MIN_VALUE%10)){
                ans=0;
                break;
            }
            ans=ans*10+temp;
        }
        return ans;
    }

    public String convert(String s, int numRows) {
        if (numRows==1)return s;
        List<StringBuilder> row=new ArrayList<>();
        for (int i = 0; i< Math.min(numRows,s.length()); i++){
            row.add(new StringBuilder());
        }
        int cur=0;
        boolean flag=false;
        for (char c:s.toCharArray()){
            row.get(cur).append(c);
            if (cur==0||cur==numRows-1)flag=!flag;
            cur+=flag?1:-1;
        }
        StringBuilder sb=new StringBuilder();
        for (StringBuilder r:row)sb.append(r);
        return sb.toString();
    }

    public boolean isInterleave(String s1, String s2, String s3) {
        int n=s1.length(),m=s2.length(),t=s3.length();
        if (n+m!=t)return false;
        boolean[][] f=new boolean[n+1][m+1];
        f[0][0]=true;
        for (int i=0;i<=n;i++){
            for (int j=0;j<=m;j++){
                int p=i+j-1;
                if (i>0){
                    f[i][j]=f[i][j]||(f[i-1][j]&&s1.charAt(i-1)==s3.charAt(p));
                }
                if (j>0){
                    f[i][j]=f[i][j]||(f[i][j-1]&&s2.charAt(j-1)==s3.charAt(p));
                }
            }
        }
        return f[n][m];
    }

    public int myAtoi(String str) {
        str= str.trim();
        int n=str.length();
        int index=0;
        if (n==index)return 0;
        char[] chars=str.toCharArray();
        boolean negative=true;
        if (chars[index]=='+')index++;
        else if (chars[index]=='-'){
            index++;
            negative=false;
        }else if (!Character.isDigit(chars[index])){
            return 0;
        }
        long ans=0;
        while (index<n&& Character.isDigit(chars[index])){
            int digit=chars[index]-'0';
            if (ans>(Integer.MAX_VALUE-digit)/10){
                return negative? Integer.MAX_VALUE: Integer.MIN_VALUE;
            }
            ans=ans*10+digit;
            index++;
        }
        return (int)(negative?ans:-ans);
    }

    public int maxArea(int[] height) {
        int l=0,r=height.length-1,ans= Integer.MIN_VALUE;
        while (l<r){
            ans= Math.max((r-l)* Math.min(height[l],height[r]),ans);
            if (height[l]<height[r]){
                l++;
            }else {
                r--;
            }
        }
        return ans;
    }

    public int romanToInt(String s) {
        HashMap<Character, Integer> map=new HashMap<>();
        map.put('I',1);
        map.put('V',5);
        map.put('X',10);
        map.put('L',50);
        map.put('C',100);
        map.put('D',500);
        map.put('M',1000);
        int sum=0;
        int pre=map.get(s.charAt(0));
        for (int i=1;i<s.length();i++){
            int cur=map.get(s.charAt(i));
            sum+=pre<cur?-pre:pre;
            pre=cur;
        }
        sum+=pre;
        return sum;
    }

    public int[] twoSum(int[] numbers, int target) {
        int left=0,right=numbers.length-1;
        while (left<right){
            if (numbers[left]+numbers[right]==target){
                return new int[]{left+1,right+1};
            }else if (numbers[left]+numbers[right]<target){
                left++;
            }else right--;
        }
        return new int[]{-1,-1};
    }
}
