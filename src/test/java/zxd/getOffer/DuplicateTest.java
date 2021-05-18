package zxd.getOffer;

import org.junit.Assert;
import org.junit.Test;

public class DuplicateTest {

    private int[] numbers = new int[]{4,3,1,0,2,5,6};
    FindDuplicateNumber duplicate = new FindDuplicateNumber();

    @Test
    public void duplicate() {
        Assert.assertEquals(2, duplicate.duplicate(numbers));
    }
}