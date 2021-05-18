package zxd.getOffer;

import org.junit.Test;

import static org.junit.Assert.*;

public class StudentTest {

    @Test
    public void test(){
        int id = 1;
        String name = "lsq";
        String sex = "female";
        Student stu = new Student();
/*        stu.setId(id);
        stu.setName(name);
        stu.setSex(sex);*/
        System.out.println(stu.toString());
    }
}