package test;

import java.util.*;
import java.util.stream.Collectors;

class Solution {
    public static int[] smallestK(int[] arr, int k) {
PriorityQueue<Integer> stack=new PriorityQueue<>(Comparator.reverseOrder());
        for (int a:arr){
            stack.add(a);
            if (stack.size()>k){
                stack.poll();
            }
        }
        return stack.stream().mapToInt(Integer::intValue).toArray();
    }

    public static void main(String[] args) {
        int[] arr={1,3,5,7,2,4,6,8};
        List<Integer> list = Arrays.stream(arr).boxed().collect(Collectors.toList());
        Collections.sort(list,(o1, o2) -> o2-o1);
        System.out.println(list);
    }
}