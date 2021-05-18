package zxd.getOffer;

import com.sun.jmx.snmp.Timestamp;
import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.*;

public class PracticeTest {

    @Test
    public void twoSum() {
        int[] nums = new int[]{2,9,10,6,7};
        int target = 13;
        int[] res = Practice.twoSum(nums, target);
        for (int i = 0; i < res.length; i++) {
            System.out.print(res[i] + " ");
        }
        System.out.println();
    }
}