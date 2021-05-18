package zxd.getOffer;

import java.util.*;
import java.lang.*;

public class Solution {
    /*
     * 链表反转，用栈*/
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode == null) return null;
        ArrayList<Integer> array = new ArrayList<Integer>();
        Stack<ListNode> stack = new Stack<ListNode>();
        while (listNode != null) {
            stack.push(listNode);
            listNode = listNode.next;
        }
        while (!stack.empty()) {
            array.add(stack.pop().val);
        }
        return array;
    }

    //不用stack
    public ArrayList<Integer> printListFromTailToHead2(ListNode listNode) {
        ArrayList<Integer> array = new ArrayList<Integer>();
        listNode = reverseList(listNode);
        while (listNode != null) {
            array.add(listNode.val);
            listNode = listNode.next;
        }
        return array;
    }

    //链表反转
    public ListNode reverseList(ListNode head) {
        if (head == null && head.next == null) return head;
        ListNode pre = null;
        ListNode next = null;
        while (head != null) {
            //保留下一个节点   
            next = head.next;
            //当前节点的下一个节点为其前一个节点
            head.next = pre;
            //当前节点为下一个节点的前一个节点
            pre = head;
            //当前节点指向下一个节点
            head = next;
        }
        return pre;
    }

    public int preIndex = 0;
    public int inIndex = 0;


    //重建二叉树
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
//        TreeNode root=reConstructBinaryTree(pre,0,pre.length-1,in,0,in.length-1);
        TreeNode root = reConstructBinaryTree(pre, in, null);
        return root;
    }

    public TreeNode reConstructBinaryTree(int[] pre, int[] in, TreeNode finish) {
        if (preIndex == pre.length || (finish != null && in[inIndex] == finish.val)) {
            return null;
        }
        //前序遍历 根左右
        //当前子树的根节点
        TreeNode root = new TreeNode(pre[preIndex++]);
        //左子树
        root.left = reConstructBinaryTree(pre, in, root);
        inIndex++;
        root.right = reConstructBinaryTree(pre, in, finish);
        return root;
    }

    //前序遍历{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}
    private TreeNode reConstructBinaryTree(int[] pre, int startPre, int endPre, int[] in, int startIn, int endIn) {

        if (startPre > endPre || startIn > endIn)
            return null;
        TreeNode root = new TreeNode(pre[startPre]);

        for (int i = startIn; i <= endIn; i++)
            if (in[i] == pre[startPre]) {
                root.left = reConstructBinaryTree(pre, startPre + 1, startPre + i - startIn, in, startIn, i - 1);
                root.right = reConstructBinaryTree(pre, i - startIn + startPre + 1, endPre, in, i + 1, endIn);
                break;
            }
        return root;
    }

    /*
     * 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行*/
    //广度优先搜索
    public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
        //取一个队列，把树的根节点放进去
        Queue<TreeNode> queue = new LinkedList<>();
        if (root != null) {
            queue.add(root);
        }
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        int j = 0;
        while (!queue.isEmpty()) {
            ArrayList<Integer> tmp = new ArrayList<>();
            //每一层循环一次
            for (int i = queue.size() - 1; i >= 0; i--) {
                //从队列里取出一个节点,放进
                TreeNode node = queue.poll();
                tmp.add(node.val);
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            j++;
            if (j % 2 == 0) {
                Collections.reverse(tmp);
            }
            res.add(tmp);

        }
        return res;
    }


    /*
     * 给定一棵二叉搜索树，请找出其中第k大的节点。*/
    public int k;
    public TreeNode node;

    public TreeNode KthNode(TreeNode pRoot, int k) {
        if (k <= 0) return null;
        this.k = k;
        dfs(pRoot);
        return node;
    }

    /*二叉搜索树，根据左根右，得到的是从小到大的序列。
     * 如果是右根左，得到的是从大到小的*/
    private void dfs(TreeNode pRoot) {
        if (pRoot == null) return;
        dfs(pRoot.right);
        if (k == 0) return;
        //递归到最左下的点，该结点为顺数第一个，要减一
        if (--k == 0)
            this.node = pRoot;  //此处不能直接return，因为return还会回到前一个节点，即pRoot保留的值不是当前的值，二是最前的一个Proo的值，即根节点；所以需要一个node保留当前的结点
        dfs(pRoot.left);
    }


    public String rserialize(TreeNode root, String str) {
        if (root == null) {
            str += "None,";
        } else {
            str += str.valueOf(root.val) + ",";
            str = rserialize(root.left, str);
            str = rserialize(root.right, str);
        }
        return str;
    }

    public String serialize(TreeNode root) {
        return rserialize(root, "");
    }

    public TreeNode rdeserialize(List<String> l) {
        if (l.get(0).equals("None")) {
            l.remove(0);
            return null;
        }

        TreeNode root = new TreeNode(Integer.valueOf(l.get(0)));
        l.remove(0);
        root.left = rdeserialize(l);
        root.right = rdeserialize(l);

        return root;
    }

    public TreeNode deserialize(String data) {
        String[] data_array = data.split(",");
        List<String> data_list = new LinkedList<String>(Arrays.asList(data_array));
        return rdeserialize(data_list);
    }

    /*判断二叉树是不是堆成二叉树*/
    public boolean isSymmetrical(TreeNode pRoot) {
        if (pRoot == null) return true;
        return recur(pRoot.left, pRoot.right);
    }

    boolean recur(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left == null || right == null || (left.val != right.val)) return false;
        return recur(left.left, right.right) && recur(left.right, right.left);
    }

    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        TreeLinkNode node = null;
        if (pNode == null) return null;
        //1.如果有右子树，找右子树最左的左子树
        if (pNode.right != null) {
            node = pNode.right;
            while (node.left != null) node = node.left;
            return node;
        }
        //2.没有右子树，找“第一个”当前节点是它的父节点的左孩子的结点，它的父节点就是下一个中序遍历的结点
        while (pNode.next != null) {
            if (pNode.next.left == pNode) return pNode.next;
            pNode = pNode.next;
        }
        return node;
    }

    /*
     * 树的深度*/
    /*
     * 即深度优先搜索 dfs*/
    public int TreeDepth(TreeNode root) {
        if (root == null) return 0;
        //算法；深度 = max(左子树深度，右子树深度) + 1
        int left = TreeDepth(root.left);
        int right = TreeDepth(root.right);
        return Math.max(left + 1, right + 1);
    }

    /*判断是不是平衡二叉树*/
    public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null) return true;
        return dfs2(root) != -1;
    }

    /*思路：从底向上*/
    public int dfs2(TreeNode root) {
        if (root == null) return 0;
        //1.求左子树深度
        int left = dfs2(root.left);
        //2.如果左子树深度为-1，即不是二叉搜索树
        if (left == -1) return -1;
        int right = dfs2(root.right);
        if (right == -1) return -1;
        //3.因为最左的子树深度 - 最右子树的深度 < 2 的时候是平衡二叉树，就继续往下，找到最深的深度
        return Math.abs(left - right) < 2 ? Math.max(left + 1, right + 1) : -1;
    }

    /*输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。
    从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。
     */
    public ArrayList<ArrayList<Integer>> res = new ArrayList<>();
    public ArrayList<Integer> path = new ArrayList<>();

    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        recur(root, target);
        return res;
    }

    void recur(TreeNode root, int target) {
        if (root == null) return;
        path.add(root.val);
        target = target - root.val;
        if (target == 0 && root.left == null && root.right == null) {
            //这里不可以res.add(path); 因为res添加相同的类时会覆盖原来的
//            res.add(path);
            res.add(new ArrayList<>(path));
        }
        recur(root.left, target);
        recur(root.right, target);
        path.remove(path.size() - 1);
    }


    /*输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

    B是A的子结构， 即 A中有出现和B相同的结构和节点值。*/
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (A == null || B == null) return false;
        //如果当前的A的根节点就是B的根节点就直接递归找下去，否则找左子树，右子树; 并且||有阻断作用
        return isSubStructureRecur(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }

    boolean isSubStructureRecur(TreeNode A, TreeNode B) {
        //只要B树指向空，说明B树已经完成了匹配
        if (B == null) return true;
        if (A == null || A.val != B.val) return false;
        //前面都走通，说明匹配了，需要匹配左右子树
        return isSubStructureRecur(A.left, B.left) && isSubStructureRecur(A.right, B.right);
    }

    public int MoreThanHalfNum_Solution(int[] array) {
        HashMap<Integer, Integer> hash = new HashMap<>();
        for (int i : array) {
            if (hash.get(i) == null) hash.put(i, 1);
            else {
                hash.put(i, hash.get(i) + 1);
            }
        }
        int result = 0;
        Set<Map.Entry<Integer, Integer>> entries = hash.entrySet();
        for (Map.Entry<Integer, Integer> entry : entries) {
            if (entry.getValue() > array.length / 2) {
                result = entry.getKey();
                break;
            }
        }
        return result;
    }

    public int majorityElement(int[] nums) {
        if (nums == null) return -1;
        int vote = 0, nowNum = nums[0];
        for (int num : nums) {
            if (vote == 0) nowNum = num;
            vote = vote + (nowNum == num ? 1 : -1);
        }
        int count = 0;
        for (int i : nums) {
            if (nowNum == i) count++;
            if (count > nums.length / 2) break;
        }
        return nowNum;
    }

    /*
    给定一棵二叉搜索树，请找出其中第k大的节点。
    * */
    public int result = 0;
    public int kNode;

    public int kthLargest(TreeNode root, int k) {
        //二叉搜索树是左根右，顺序是从小到大，如果是右根左，就是从大到小
        this.kNode = k;
        recur1(root);
        return result;
    }

    private void recur1(TreeNode root) {
        if (root == null) return;
        recur1(root.right);
        if (kNode == 0) return;
        //当k为1时，也就是找到要找的结点
        if (kNode == 1) result = root.val;
        kNode--;
        recur1(root.left);
    }

    /*两数之和*/
    public int[] twoSum(int[] numbers, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        int[] res = new int[2];
        if (numbers == null) return res;
        for (int i = 0; i < numbers.length; i++) {
            //1.数字加入map之前，判断map是否存在key为target - numbers[i]的值，如果有说明存在值
            //2.相同值的解释：因为题目只存在一种组合，所以相同值必是结果，因为无法有三个相同值为结果的情况
            if (map.containsKey(target - numbers[i])) {
                res[0] = map.get(target - numbers[i]);
                res[1] = i + 1;
                break;
            }
            map.put(numbers[i], i + 1);
        }
        return res;
    }

    /*
     * 排序链表
     * 时间：O(nlogn)
     * 空间: O(1)*/
    public ListNode sortList(ListNode head) {
        //1.判空
        if (head == null || head.next == null) {
            return head;
        }
        //2.对半拆分; 方法：快慢指针
        ListNode slow = head, fast = head.next;
        //后面的fast.next != null的判断，是对应只有两个节点的情况
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        //循环完毕，slow指针的位置在：偶数节点，位于n/2的位置；奇数节点，位于n/2 + 1节点
        //slow指针下一个节点，为二分两外一半的头结点
        ListNode tmp = slow.next;
        slow.next = null;
        //递归划分
        ListNode left = sortList(head);
        ListNode right = sortList(tmp);

        //合并链表
        ListNode back = new ListNode(0), res = back;

        while (left != null && right != null) {
            if (left.val < right.val) {
                back.next = left;
                left = left.next;
            } else {
                back.next = right;
                right = right.next;
            }
            back = back.next;
        }
        back.next = (left != null ? left : right);
        return res.next;
    }

    /*排序k个链表*/
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> q = new PriorityQueue<>((x, y) -> x.val - y.val);
/*        PriorityQueue<ListNode> q1 = new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val - o2.val;
            }
        });*/

        for (ListNode node : lists) {
            if (node != null) {
                q.add(node);
            }
        }
        ListNode head = new ListNode(0);
        ListNode tail = head;
        while (!q.isEmpty()) {
            tail.next = q.poll();
            tail = tail.next;
            if (tail.next != null) {
                q.add(tail.next);
            }
        }
        return head.next;
    }



    public int removeDuplicates(int[] nums) {
        return process(nums, 2);
    }

    private int process(int[] nums, int k) {
        int u = 0;
        int i = 0;
        for (int x : nums) {
            if (u < 2 || nums[u - 2] != x) nums[u++] = x;
            i++;
        }
        return u;
    }


    public boolean isMatch(String s, String p) {
        int m = s.length() + 1, n = p.length() + 1;
        boolean[][] dp = new boolean[m][n];
        dp[0][0] = true;
        // 初始化首行
        for (int j = 2; j < n; j += 2)
            dp[0][j] = dp[0][j - 2] && p.charAt(j - 1) == '*';
        // 状态转移
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (p.charAt(j - 1) == '*') {
                    if (dp[i][j - 2]) dp[i][j] = true;                                            // 1.
                    else if (dp[i - 1][j] && s.charAt(i - 1) == p.charAt(j - 2)) dp[i][j] = true; // 2.
                    else if (dp[i - 1][j] && p.charAt(j - 2) == '.') dp[i][j] = true;             // 3.
                } else {
                    if (dp[i - 1][j - 1] && s.charAt(i - 1) == p.charAt(j - 1)) dp[i][j] = true;  // 1.
                    else if (dp[i - 1][j - 1] && p.charAt(j - 1) == '.') dp[i][j] = true;         // 2.
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    public int longestCommonSubsequence(String text1, String text2) {
        //对于两个字符串的题目，用两个指针分别对应两个字符串
        int m = text1.length() + 1, n = text2.length() + 1;

        int[][] dp = new int[m][n];
        //当两个字符串为空的时候，表示两个的公共子序列为0
        dp[0][0] = 0;
        for (int i = 0; i < m; i++) {
            dp[m][0] = 0;
        }
        for (int i = 0; i < n; i++) {
            dp[0][n] = 0;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = (dp[i - 1][j] > dp[i][j - 1] ? dp[i - 1][j] : dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }

    public int lengthOfLIS(int[] nums) {
        int length = nums.length;
        int res = 1;
        //状态是轮到这个数字的时候，对应的最长递增子序列是dp[i]
        //判断是比前面最大的一个元素对应的数组元素还大，即nums[i+1] > nums[i]时，dp[i] + 1;
        int[] dp = new int[length];
        for (int i = 0; i < length; i++) {
            dp[i] = 1;
        }
        //让当前下标为i的数字，逐个与前面的比较，并更新当前自己的dp[i]
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) dp[i] = Math.max(dp[i], dp[j] + 1);
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    /*
     * 反转String里带空格的部分
     * */
    public String reverseWords(String s) {
        char[] chars = s.toCharArray();
        //用一个值去判断当前的为空格
        int j = 0;
        for (int i = 0; i < chars.length; i++) {
            if (chars[i] == ' ') {
                reverse(chars, i - 1, j);
                j = i + 1;
            }
        }
        //最后一个循环的单词
        reverse(chars, chars.length - 1, j);
        return new String(chars);
    }

    private void reverse(char[] chars, int i, int j) {
        while (j < i) {
            char temp = chars[i];
            chars[i] = chars[j];
            chars[j] = temp;
            i--;
            j++;
        }
    }

    public int maxProfit(int[] prices) {
        int[] dp = new int[prices.length]; //默认每一天的收益最少是0元
        int res = 0;
        for (int i = 1; i < dp.length; i++) {
            for (int j = 0; j < i; j++) {
                if (prices[i] > prices[j]) {
                    dp[i] = Math.max(dp[i], prices[i] - prices[j]);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    //买卖股票的最佳时机2
    public int maxProfit2(int[] prices) {
        //
        /*
         profit[i][0] : 第i天交易结束后不持有股票的时候最大收益
         什么状态可以导致这个结果：
        （1）profit[i-1][1] + prices[i] 昨天交易结束的时候有股票，今天卖出，卖出后收益提升
        （2）profit[i-1][0] 今天不买股票，那今天的收益就是昨天的买卖股票的收益

         profit[i][1] : 第i天交易结束后持有股票的时候最大收益
         什么状态可以导致这个结果：
         （1）profit[i-1][0] - prices[i] 昨天交易的时候没有持有股票，今天买入
         （2）profit[i-1][1]  昨天有股票，今天不操作
        * */
        //int[i][1] : 第i天交易结束后持有股票的时候最大收益
        int[][] profit = new int[prices.length][2];
        //第一天没有持有股票，收益为0
        profit[0][0] = 0;
        //第一天持有股票，收益为买当天股票的消耗的钱
        profit[0][1] = -prices[0];
        for (int i = 1; i < prices.length; i++) {
            profit[i][0] = Math.max(profit[i - 1][0], profit[i - 1][1] + prices[i]);
            profit[i][1] = Math.max(profit[i - 1][1], profit[i - 1][0] - prices[i]);
        }
        return profit[prices.length - 1][0];
    }

    /*括号匹配*/
    public boolean isValid(String s) {
        if (s == null || s.length() == 0) return false;
        int length = s.length();
        //如果是奇数就
        if (length % 2 == 1) return false;
        Stack<Character> stack = new Stack<>();
        stack.push(s.charAt(0));
        for (int i = 1; i < length; i++) {
            //当前的字符和栈顶的字符比较
            char left = stack.peek();
            char right = s.charAt(i);
            if (right == ')' && left == '(' ||
                    right == ']' && left == '[' ||
                    right == '}' && left == '{') {
                stack.pop();
                continue;
            }
            stack.push(right);
        }
        return stack.isEmpty();
    }

    /*整数反转*/
    public int reverseNumber(int x) {
        //Math.MAX_VALUE = 2147483647  , 负数为 -2147483648
        //如果使用res = res*10 + num
        //如果num是正数，则res > Math.MAX_VALUE/10时，比如是214748365 *10 后必大于最大值
        //如果num是负数  则res < Math.MAX_VALUE/10，比如.... 后必小于最小值
        //倘若是=的情况，则比较最后一位如，2147483640的时候，如果最后一位加的数字大于7必溢出；负数情况就是小于-8的时候
        int maxNum = Integer.MAX_VALUE;
        int minNum = Integer.MIN_VALUE;
        int num = 0, res = 0;
        while (x != 0) {
            num = x % 10;
            x /= 10;
            //正数 2147483647
            if (res > maxNum / 10 || (res == maxNum / 10 && num > 7)) return 0;
            //负数 -2147483648
            if (res < minNum / 10 || (res == minNum / 10 && num < -8)) return 0;
            res = res * 10 + num;
        }
        return res;
    }

    /*
     * 回文数
     * */
    public boolean isPalindrome(int x) {
        String str = String.valueOf(x);
        return false;
    }

    public String longestCommonPrefix(String[] strs) {
        /*
        1. 用一个字符串初始化赋值为strs[0]
        2. 用str逐个和strs[i]里的比较
        */
        if (strs == null || strs.length == 0) return "";
        String ans = strs[0];
        for (int i = 0; i < strs.length; i++) {
            //如果第一个字符就不等，或者某一个字符串是"",那就
            if (strs[i].length() == 0 || ans.charAt(0) != strs[i].charAt(0)) return "";
            //res用来记录每一次ans个strs[i]的公共子前序
            String res = "";
            //将ans字符串逐个和字符串数组里的字符串比较，需要注意的是，因为未知ans字符串的长度和strs[i]的长度谁长，
            // 所以以最短的为循环结束表示
            for (int j = 0; j < ans.length() && j < strs[i].length(); j++) {
                //如果相等就将结果加到新的字符串中
                if (ans.charAt(j) == strs[i].charAt(j)) {
                    res += ans.charAt(j);
                    continue;
                } else {  //如果不等，可以跳出循环
                    break;
                }
            }
            //循环最后更新ans为当次比较的结果
            ans = res;
        }
        return ans;
    }

    public String longestCommonPrefix2(String[] strs) {
        //边界条件判断
        if (strs == null || strs.length == 0) return "";

        String ans = strs[0];
        for (int i = 1; i < strs.length; i++) {
            if (ans.length() == 0) return "";
            //如果ans是strs[i]的子串，返回子串重合的strs[i]的第一个index
            while (strs[i].indexOf(ans) != 0) {
                //将ans去掉最后一个字符，如果一个都不符合，那ans就是""
                ans = ans.substring(0, ans.length() - 1);
            }
        }
        return ans;
    }

    /*返回最近公共子节点*/
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        //递归到最后叶子节点往后的null或者遇到p，q，直接返回root
        if (root == null || root.val == p.val || root.val == q.val) return root;
        //后序遍历
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        //1.左右子树都没有找到p，q，说明这一边的树没有值
        if (left == null && right == null) return null;
        //2.左子树为空，找右子树
        if (left == null) return right;
        //3.右子树为空，找左子树
        if (right == null) return left;
        //最终如果左右子树都不为空
        return root;
    }

    /*
     * 160. 相交链表*/
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode A = headA, B = headB;
        //如果A == B，说明两个人都指向了同一个地址
        //如果两个人找到相等的，两种情况，两个人找到节点了，或者两个人都是空的
        while (A != B) {
            //1.分别循环A，然后再循环A-C的部分，即没有重合的B的部分;B同理
            A = (A != null ? A.next : headB);
            B = (B != null ? B.next : headA);
        }
        return A;
    }

    /*
     * 160. 相交链表*/
    public ListNode getIntersectionNode2(ListNode headA, ListNode headB) {
        Map<ListNode, Integer> map = new HashMap<ListNode, Integer>();
        while (headA != null) {
            map.put(headA, headA.val);
            headA = headA.next;
        }
        while (headB != null) {
            if (map.containsKey(headB)) return headB;
            headB = headB.next;
        }
        return null;
    }

    /*217. 存在重复元素*/
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num)) return true;
            set.add(num);
        }

        return false;
    }

    /*70. 爬楼梯*/
    public int climbStairs(int n) {
/*      状态变换的是每一阶楼梯有多少种上去的方法
        规律  n = 1   res = 1
              n = 2   res = 2
              n = 3   res = 3
              n = 4   res = 5
         第n层 =  res(n-1) + res(n - 2)
 */
        //第0层初始化为1，可以理解为别的层都是要往上走，而第0层只有一种方式就是原地踏步
        int n0 = 1;
        int n1 = 1;
        for (int i = 2; i <= n; i++) {
            int temp = n1;
            n1 = n0 + n1;
            n0 = temp;
        }
        return n1;
    }

    /*141. 环形链表*/
    public boolean hasCycle(ListNode head) {
        //快慢指针，慢指针一次走一个节点，快指针一次性走两个节点
        if (head == null || head.next == null) return false;
        ListNode slow = head, fast = head;
        while (slow != null && fast != null) {
            slow = slow.next;
            if (fast.next != null)
                //要小心fast.next可能是null的时候，执行fast.next.next会报空指针的错误
                fast = fast.next.next;
            else return false;
            if (slow == fast) return true;
        }
        return false;
    }

    /*88. 合并两个有序数组*/
    public void merge(int[] nums1, int m, int[] nums2, int n) {

        //常规做法是先把nums2赋值到nums1中，然后再排序
        //快一点的方法就是用额外开一个数组，用双指针从头到尾判断数组大小，最后得出结果

        //巧妙的做法，平时都是从下标比较小的开始，接下来的可以从下标大的开始，两个数组元素比较，谁最大，
        //谁就放在数组的末尾

        //谁比较大就放置在这里
        if (n == 0) return;
        //双指针
        int i1 = m - 1, i2 = n - 1;
        if (m > 0) {
            for (int maxIndex = m + n - 1; maxIndex >= 0; maxIndex--) {
                if (nums1[i1] > nums2[i2]) {
                    nums1[maxIndex] = nums1[i1];
                    if (--i1 < 0) break;
                } else {
                    nums1[maxIndex] = nums2[i2];
                    if (--i2 < 0) break;
                }
            }
        }
        for (int i = i2; i >= 0; i--) {
            nums1[i] = nums2[i];
        }
    }

    /*231. 2的幂*/
    public boolean isPowerOfTwo(int n) {
        //n为不为1的奇数
        if (n % 2 == 1 && n != 1 || n <= 0) return false;

        while (n % 2 != 1) {
            n = n / 2;
        }
        return n == 1;
    }

    /*5. 最长回文子串*/
    /*中心扩散的方法，分为奇数扩散和偶数扩散如    abcba从c开始的奇数扩散，  abba bb开始的偶数扩散*/
    public String longestPalindrome(String s) {
        //如果为空，直接返回s
        if (s.length() == 0) return s;
        //初始化最长为1
        int res = 1;
        //两个指针作为最终的左右指针
        int left = 0;
        int right = s.length() - 1;
        for (int i = 0; i < s.length(); i++) {
            //判断奇数的情况
            //此时i是中心节点，声明两个i的左右节点
            int li = i - 1;
            int ri = i + 1;
            //下标越界情况要排除
            while (li >= 0 && ri <= s.length() - 1 && s.charAt(li) == s.charAt(ri)) {
                //此时的回文串的长度
                int len = ri - li + 1;
                //更新长度和左右下标的值
                if (len > res) {
                    res = len;
                    left = li;
                    right = ri;
                }
                //向左右扩散
                li--;
                ri++;
            }

            //判断偶数的情况
            li = i;
            ri = i + 1;
            while (li >= 0 && ri <= s.length() - 1 && s.charAt(li) == s.charAt(ri)) {
                //此时的回文串的长度
                int len = ri - li + 1;
                //更新长度和左右下标的值
                if (len > res) {
                    res = len;
                    left = li;
                    right = ri;
                }
                //向左右扩散
                li--;
                ri++;
            }
        }
        return s.substring(left, right + 1);
    }

    /*11. 盛最多水的容器*/
    public int maxArea(int[] height) {
        //双指针法，一个指向头部，一个尾部；保证了起码在地面的面积最大
        //如何移动？
        //谁的高度比较小就移动谁
        int res = 0, area = 0;
        int left = 0, right = height.length - 1;
        while (left != right) {
            res = height[left] < height[right] ?
                    Math.max(height[left++] * (right - left), res) :
                    Math.max(height[right--] * (right - left), res);
        }
        return res;
    }

    /*15. 三数之和*/
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new LinkedList<>();
        if (nums.length < 3) return result;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                break;
            }
            //如果当前的数和前面一个数相等，代表上一轮循环已经执行过了
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int threeSum = nums[i] + nums[left] + nums[right];
                if (threeSum < 0) {
                    left++;
                } else if (threeSum > 0) {
                    right--;
                } else {
                    LinkedList<Integer> res = new LinkedList<>();
                    res.add(nums[i]);
                    res.add(nums[left]);
                    res.add(nums[right]);
                    result.add(res);
                    while (left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    while (left < right && nums[right] == nums[right - 1]) {
                        right--;
                    }
                    right--;
                    left++;
                }
            }
        }
        return result;
    }

    /*8. 字符串转换整数 (atoi)*/
    /*
     * 1. 第一个字符是空格就往下，第一个要的字符是'-'或者是数字,结束是第一个非数字的
     * */
    public int myAtoi(String s) {
        return 0;
    }

    /*46. 全排列*/
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();

        List<Integer> output = new ArrayList<Integer>();
        for (int num : nums) {
            output.add(num);
        }

        int n = nums.length;
        backtrack(n, output, res, 0);
        return res;
    }

    public void backtrack(int n, List<Integer> output, List<List<Integer>> res, int first) {
        // 所有数都填完了
        if (first == n) {
            res.add(new ArrayList<Integer>(output));
        }
        for (int i = first; i < n; i++) {
            // 动态维护数组
            Collections.swap(output, first, i);
            // 继续递归填下一个数
            backtrack(n, output, res, first + 1);
            // 撤销操作
            Collections.swap(output, first, i);
        }
    }

    /*142. 环形链表 II*/
    public ListNode detectCycle(ListNode head) {
        //快慢指针，如果有环，最终两个链表必能走到一块
        if (head == null || head.next == null) {
            return head;
        }
        ListNode slow = head, fast = head;
        while (true) {
            //只需判断快指针的就可以了，因为快指针不出现空指针的情况的话，慢指针也不可能会出现
            if (fast == null || fast == fast.next) return null;
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                break;
            }
        }
        fast = head;
        while (fast != slow) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    /*61. 旋转链表*/
    public ListNode rotateRight(ListNode head, int k) {

        ListNode back = head;
        int count = 0;
        while (back != null) {
            back = back.next;
            count++;
        }
        back = head;
        int nodeIndex = k % count;
        if (nodeIndex == 0) return head;
        //找导数第nodeIndex个节点
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode pre = dummy;
        while (nodeIndex != 0) {
            back = back.next;
            nodeIndex--;
        }
        pre = head;
        while (back.next != null) {
            pre = pre.next;
            back = back.next;
        }
        back.next = head;
        head = pre.next;
        pre.next = null;
        return head;
    }


    public int[] twoSum1(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], i);
        }
        Arrays.sort(nums);
        int left = 0;
        int right = nums.length - 1;
        int[] res =  new int[2];
        while (left < right) {
            int sum = nums[left] + nums[right];
            if (sum > target) {
                right--;
            } else if (sum < target) {
                left++;
            } else {
                res[0] = map.get(nums[left]);
                res[1] = map.get(nums[right]);
                return res;
            }
        }
        return res;
    }
    public int[] twoSum2(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{i, map.get(target - nums[i])};
            } else {
                map.put(target - nums[i], i);
            }
        }
        return new int[2];
    }




}
