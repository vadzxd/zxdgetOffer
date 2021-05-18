package zxd.getOffer;

import org.junit.Assert;
import org.junit.Test;

import java.util.*;

public class SolutionTest {
    private int [] pre = new int[]{3, 9, 8, 5, 4, 10, 20, 15, 7};
    private int [] in = new int[]{4, 5, 8, 10, 9, 3, 15, 20, 7};
    private Integer [] test = new Integer[]{1,2,3};
    Solution solution = new Solution();
    TreeNode node = TreeNode.constructTree(test);
    private int [] aaa = new int[]{1,2,3,4,5};
    ListNode listNode = ListNode.createLinkedList(aaa);

    @Test
    public void kthNode() {
        System.out.println(solution.KthNode(node,1).val);
    }



    @Test
    public void reConstructBinaryTree() {
        TreeNode treeNode = solution.reConstructBinaryTree(pre,in,null);
        TreeNode.preOrder(treeNode);

    }

    @Test
    public void levelOrder() {
        ArrayList<ArrayList<Integer>> list = solution.levelOrder(node);

        for (int i = 0; i < list.size(); i++) {
            System.out.println(list.get(i));
        }
    }

    @Test
    public void constructBinaryTree(){
        TreeNode.preOrder(TreeNode.constructTree(test));
    }


    @Test
    public void serialize() {
        System.out.println(solution.serialize(node));

    }

    @Test
    public void isSymmetrical() {
        System.out.println(solution.isSymmetrical(node));
    }

    @Test
    public void isBalanced_Solution() {
        System.out.println(solution.IsBalanced_Solution(node));
    }

    @Test
    public void findPath() {
        ArrayList<ArrayList<Integer>> list = solution.FindPath(node,22);
        for (int i = 0; i < list.size(); i++) {
            System.out.println(list.get(i));
        }
    }

    @Test
    public void moreThanHalfNum_Solution() {
        int[] ints = new int[]{1,2,3,2,2,2,5,4,2};
        Assert.assertEquals(2,solution.MoreThanHalfNum_Solution(ints));
    }

    @Test
    public void majorityElement() {
        int[] ints = new int[]{1,2,1,1,4};
        Assert.assertEquals(1,solution.majorityElement(ints));
    }

    @Test
    public void kthLargest() {
        System.out.println(solution.kthLargest(node,1));
    }

    @Test
    public void testFunction(){
        Integer[] numbers = new Integer[]{1,2,5,2,4};
        Arrays.sort(numbers);
        for (int number : numbers) {
            System.out.print(number + " ");
        }
        System.out.println();
        Arrays.sort(numbers, Collections.reverseOrder());
        for (int number : numbers) {
            System.out.print(number + " ");
        }
    }

    @Test
    public void twoSum() {
        int[] numbers = new int[]{1,2,5,2,4};
        int target = 4;
        for (int i : solution.twoSum(numbers, target)) {
            System.out.print(i + " ");
        }


    }

    @Test
    public void sortList() {
        ListNode listNode1 = solution.sortList(this.listNode);
        for (int i : pre) {
            System.out.print(i + ",");
        }
        System.out.println();
        while (listNode1 != null) {
            System.out.print(listNode1.val + ",");
            listNode1 = listNode1.next;
        }
    }

    @Test
    public void priorityQueueTest() {
        PriorityQueue<Integer> queue = new PriorityQueue<>((x,y) -> x - y);
        for (int i : pre) {
            System.out.print(i + ",");
            queue.add(i);
        }
        System.out.println();
        for (Integer integer : queue) {
            System.out.print( integer + ",");
        }
        System.out.println();
        while (!queue.isEmpty()) {
            Integer in = queue.remove();
            System.out.print(in + ",");
        }
    }

    @Test
    public void fun(){
        int[] scores = {1, 20, 30, 40, 50};
        //在数组scores中查找元素20
        int res = Arrays.binarySearch(scores, 31);
        //打印返回结果
        System.out.println("res = " + res);
        System.out.println(~res);
    }

    @Test
    public void removeDuplicates() {
//        int[] nums = new int[]{1,1,2,1,2,3,2};
        int[] nums = new int[]{1,1,2,2,3};
/*        int a = solution.removeDuplicates(abc);
        System.out.println(a);*/

        int i = 0;
        for (int j = 1; j < nums.length; j++) {
            if (nums[j] != nums[i]) {
                i++;
                nums[i] = nums[j];
            }
        }
        System.out.println(i + 1);
    }

    @Test
    public void fun1(){
        System.out.println(1 >> 1);

    }

    @Test
    public void lengthOfLIS() {
        int[] nums = new int[]{1,3,6,7,9,4,10,5,6};
        System.out.println(solution.lengthOfLIS(nums));
    }

    @Test
    public void reverseWords() {
        System.out.println(solution.reverseWords("Let's take LeetCode contest  "));
    }

    @Test
    public void isValid() {
        String s = "()[]{}";
        System.out.println(solution.isValid(s));
    }

    @Test
    public void reverseNumber() {
        int[] ans = new int[]{1230,-1230,2147483647,-2147483648,1463847412};
        for (int an : ans) {
            System.out.println(an + "结果： " + solution.reverseNumber(an));
        }
    }

    @Test
    public void longestCommonPrefix() {
        String[] strs = new String[]{"",""};
        System.out.println(solution.longestCommonPrefix(strs));

    }

    @Test
    public void isPowerOfTwo() {
        System.out.println(solution.isPowerOfTwo(8));
    }

    @Test
    public void longestPalindrome() {
        System.out.println(solution.longestPalindrome("ccc"));
    }

    @Test
    public void threeSum() {
        int[] nums = new int[]{-1,0,1,2,-1,-4};
        List<List<Integer>> list = solution.threeSum(nums);
        for (List<Integer> integers : list) {
            System.out.println(integers);
        }
    }

    @Test
    public void backtrack() {
        int[] nums = new int[]{1, 2, 3};
        List<List<Integer>> list = solution.permute(nums);
        for (int i = 0; i < list.size(); i++) {
            for (Integer integer : list.get(i)) {
                System.out.print(integer + " ");
            }
            System.out.println();
        }
    }

    @Test
    public void midSearch(){
        int[] nums = new int[]{1, 2, 3, 4, 5, 8, 9};
        int left = 0 , right = nums.length - 1, mid = 0;
        int target = 5;
        boolean isFind = false;
        while (left <= right) {
            mid = left + ((right - left) >> 1);
            if (nums[mid] == target) {
                isFind = true;
                System.out.println(mid);
                break;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (!isFind) {
            System.out.println("没找到");
        }
    }

    @Test
    public void rotateRight() {
        ListNode  a = solution.rotateRight(listNode,2);
        ListNode.printLinkedList(a);
    }

    @Test
    public void twoSum1() {
        int[] nums = new int[]{2,7, 11, 15};
        int target = 19;
        int[] result = solution.twoSum1(nums, target);
        for (int i : result) {
            System.out.print(i + " ");
        }
    }

    @Test
    public void listNode(){
        int[] nums = new int[]{0,1,2,3,4,5};
        ListNode head = ListNode.createLinkedList(nums);

        while (head != null) {
            System.out.println(head.val);
            head = head.next;
        }
    }

    @Test
    public void oneFun(){
        int x = 123;
        Integer num = new Integer(x);
        String sss = num.toString();
        String str = "" + x;
        System.out.println(sss);
    }
}