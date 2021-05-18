/*
* 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
* 请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。*/
package zxd.getOffer;

public class FindNumberIn2DArray {
    public boolean Find(int target, int [][] array) {
        boolean ans = false;
        //判断数组:二维判断：数组为空，数组长度为0；一维的判断：数组只有一个一维数组，长度为0
        if ((array == null || array.length == 0) || (array.length == 1 && array[0].length == 0))
            return false;

        int i = array.length - 1, j = 0; //左下角找起
        while(i >= 0 && j <= array.length - 1) {
            if(target == array[i][j]){
                ans = true;
                break;
            }
            else if(target < array[i][j]){
                i--;
            }
            else {
                j++;
            }
        }

        return ans;
    }
}
