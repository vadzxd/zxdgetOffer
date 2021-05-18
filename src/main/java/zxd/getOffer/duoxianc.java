package zxd.getOffer;


public class duoxianc {
    public static void main(String[] args) throws Exception {
        final Object obj = new Object();

        Thread t1 = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    while (true) {
                        synchronized (obj) {
                            obj.notifyAll();
                            obj.wait();
                            System.out.println(1);
                        }
                    }
                } catch (Exception e) {

                    e.printStackTrace();
                }
            }
        });

        Thread t2 = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    while (true) {
                        synchronized (obj) {
                            obj.notifyAll();
                            obj.wait();
                            System.out.println(2);
                        }
                    }
                } catch (Exception e) {

                    e.printStackTrace();
                }
            }
        });

        t1.start();
        t2.start();
    };
}

