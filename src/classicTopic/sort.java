package classicTopic;

import java.util.Arrays;

public class sort {

    public static void main(String[] args) {
        int[] arr={5,7,3,2,8};
        sort s=new sort();
        arr=s.mergeSort(arr);
        System.out.println(arr);
    }

    public void quickSort(int[] arr){
        quickSort(arr,0,arr.length-1);
    }

    public void quickSort(int[] arr,int left,int right){
        if (left>right)return;
        int base=arr[left];
        int i=left,j=right;
        while (i<j){
            while (arr[j]>=base&&i<j)j--;
            while (arr[i]<=base&&i<j)i++;
            if (i<j){
                int temp=arr[i];
                arr[i]=arr[j];
                arr[j]=temp;
            }
        }
        arr[left]=arr[i];
        arr[i]=base;
        quickSort(arr,left,i-1);
        quickSort(arr,i+1,right);
    }

    public void selectSort(int[] arr){
        int n=arr.length;
        for (int i=0;i<n;i++){
            for (int j=i+1;j<n;j++){
                if (arr[i]>arr[j]){
                    int temp=arr[i];
                    arr[i]=arr[j];
                    arr[j]=temp;
                }
            }
        }
    }

    public void bubbleSort(int[] arr){
        int n= arr.length;
        boolean flag=true;
        for (int i=0;i<n;i++){
            for (int j=0;j<n-i-1;j++){
                if (arr[j]>arr[j+1]){
                    int temp=arr[j];
                    arr[j]=arr[j+1];
                    arr[j+1]=temp;
                    flag=false;
                }
            }
            if (flag)break;
        }
    }

    public void insertSort(int[] arr){
        int n=arr.length;
        for (int i=1;i<n;i++){
            int temp=arr[i];
            int k=i-1;
            while (k>=0&&arr[k]>temp){
                arr[k+1]=arr[k];
                k--;
            }
            arr[k+1]=temp;
        }
    }

    public int[] mergeSort(int[] arr){
        if (arr.length<2)return arr;
        int mid=arr.length/2;
        int[] left= Arrays.copyOfRange(arr,0,mid);
        int[] right=Arrays.copyOfRange(arr,mid,arr.length);
        return merge(mergeSort(left),mergeSort(right));
    }

    public int[] merge(int[] left,int[] right){
        int n= left.length,m=right.length;
        int[] arr=new int[n+m];
        int i=0,j=0;
        for (int k=0;k<n+m;k++){
            if (i>=n){
                arr[k]=right[j++];
            }else if (j>=m){
                arr[k]=left[i++];
            }else if (left[i]<right[j]){
                arr[k]=left[i++];
            }else arr[k]=right[j++];
        }
        return arr;
    }


}
