package test;

public class sort {

    public static void main(String[] args) {
        int[] arr={5,4,7,1,23,76};
        select(arr);
        System.out.println(arr);
    }

    public static void bubble(int[] arr){
        int len= arr.length;
        for (int i=0;i<len-1;i++){
            boolean flag=false;
            for (int j=0;j<len-i-1;j++){
                if (arr[j]>arr[j+1]){
                    int temp=arr[j];
                    arr[j]=arr[j+1];
                    arr[j+1]=temp;
                    flag=true;
                }
            }
            if (!flag)break;
        }
    }

    public static void quick(int[] arr){
        quickSort(arr,0,arr.length-1);
    }

    public static void quickSort(int[] arr,int left,int right){
        if (left>right)return;
        int mid=arr[left];
        int i=left,j=right;
        while (i<j){
            while (i<j&&arr[j]>=mid)j--;
            while (i<j&&arr[i]<=mid)i++;
            if (i<j){
                int temp=arr[i];
                arr[i]=arr[j];
                arr[j]=temp;
            }
        }
        arr[left]=arr[i];
        arr[i]=mid;
        quickSort(arr,left,i-1);
        quickSort(arr,i+1,right);
    }

    public static void select(int[] arr){
        for (int i=0;i<arr.length;i++){
            int min=i;
            for (int j=i+1;j<arr.length;j++){
                if (arr[j]<arr[min]){
                    min=j;
                }
            }
            if (min!=i){
                int temp=arr[min];
                arr[min]=arr[i];
                arr[i]=temp;
            }
        }
    }
}
