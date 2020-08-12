package leetcode;

import java.util.List;

class Node {
    public int val;
    public List<Node> neighbors;

    public Node(int val, List<Node> neighbors) {
        this.val = val;
        this.neighbors = neighbors;
    }
}