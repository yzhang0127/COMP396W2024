import org.opencv.core.Core;

public class OpticalFlowDemo {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new OptFlow().run(/*args*/);
    }
}
