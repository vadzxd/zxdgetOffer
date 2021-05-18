/*
* 在一个长度为n的数组里的所有数字都在0到n-1的范围内。
* 数组中某些数字是重复的，但不知道有几个数字是重复的。
* 也不知道每个数字重复几次。请找出数组中第一个重复的数字。
* 例如，如果输入长度为7的数组[2,3,1,0,2,5,3]，那么对应的输出是第一个重复的数字2。没有重复的数字返回-1。
* */
package zxd.getOffer;

import java.util.HashSet;
import java.util.Set;

public class FindDuplicateNumber {
    public int duplicate (int[] numbers) {
        // write code here

        Set<Integer> set = new HashSet<Integer>();
        int ans = -1;
        for (int number : numbers) {
                //如果出现了相同的字符，那就返回
                if (!set.add(number)) return number;
        }
        return ans;
    }
}
