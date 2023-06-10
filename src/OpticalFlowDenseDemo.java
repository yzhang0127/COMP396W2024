import org.opencv.core.Core;

public class OpticalFlowDenseDemo {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new OpticalDense().run(/*args*/);
    }
}
