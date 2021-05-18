package zxd.getOffer;

import java.util.Stack;

public class Learn {
    public static void main(String[] args) {
        int[] arr = new int[]{0,1,2,3,4,5};
        ListNode head = ListNode.createLinkedList(arr);

        Stack<ListNode> stack = new Stack<>();

        while (head != null) {
            stack.push(head);
            head = head.next;
        }
        int val = -1;
        ListNode res = new ListNode(val);
        ListNode pre = res;

        while(!stack.empty()) {
            ListNode temp = stack.pop();
            pre.next = temp;
            pre = pre.next;
        }
        //pre.next = null;

        ListNode.printLinkedList(res.next);

    }

}
