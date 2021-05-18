package zxd.getOffer;

import java.util.HashMap;
import java.util.Map;

public class Practice {

    public static int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i <nums.length ; i++) {
            int minusRes = target - nums[i];
            if (map.containsKey(minusRes)) {
                res[0] = map.get(minusRes);
                res[1] = i;
                return res;
            } else {
                map.put(nums[i],i);
            }
        }


        return  res;

    }
}