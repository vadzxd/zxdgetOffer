package zxd.getOffer;

import org.junit.Test;

import java.util.Stack;

public class NodeTest {

    @Test
    public void fun() {

        Node node0 = new Node(0);
        Node node1 = new Node(1);
        Node node2 = new Node(2);
        Node node3 = new Node(3);
        Node node4 = new Node(4);
        Node node5 = new Node(5);

        node0.next = node1;
        node1.next = node2;
        node2.next = node3;
        node3.next = node4;
        node4.next = node5;

        Stack<Node> stack = new Stack<>();

        stack.push(node0);
        stack.push(node1);
        stack.push(node2);
        stack.push(node3);
        stack.push(node4);
        stack.push(node5);

        Node head = node0;


        while (head != null) {
            System.out.print(head.val + ",");
            head = head.next;
        }
    }

}