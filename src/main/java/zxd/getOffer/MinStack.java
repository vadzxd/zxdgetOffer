package zxd.getOffer;

import java.util.Stack;

public class MinStack {
    //没有什么简单的做法，用一个辅助栈和一个数据栈;辅助栈的栈顶永远都是最小的值
    private Stack<Integer> data;
    private Stack<Integer> helper;
    /** initialize your data structure here. */
    public MinStack() {
        data = new Stack<>();
        helper = new Stack<>();

    }

    public void push(int val) {
        data.push(val);
        if(helper.isEmpty() || helper.peek() > val){
            helper.push(val);
        }
        else {
            helper.push(helper.peek());
        }
    }

    public void pop() {
        data.pop();
        helper.pop();
    }

    public int top() {
        return data.peek();
    }

    public int getMin() {
        return helper.peek();
    }
}
