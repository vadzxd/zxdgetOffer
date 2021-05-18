package zxd.getOffer;

public class ReplaceSpace {
    public String replaceSpace (String s) {
        if (s == null){
            return s;
        }
        //记录s的长度
        int length = s.length();
        //一个空格变成%20，变成3倍
        char[] str = new char[length * 3];
        int size = 0;
        for (int i = 0; i < length; i++) {
            char a = s.charAt(i);
            if (a == ' '){
                str[size++] = '%';
                str[size++] = '2';
                str[size++] = '0';
            }
            else {
                str[size++] = a;
            }
        }
        String s1 = new String(str, 0, size );
        return s1;
    }
}
