import java.util.*;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.*;
import org.opencv.videoio.*;


public class ObjectAvoidance {
    public void cam_OptFlow(String filename, boolean useFile, int channel){
        VideoCapture capture = null;
        //capture video frames
        if(useFile){
            capture = new VideoCapture(filename);

        }
        else{
            //System.out.println("use "+channel+" channel");
            capture = new VideoCapture(channel);
        }

        if (!capture.isOpened()) {
            System.out.println("Unable to capture");
            System.exit(-1);
        }

        //create Video Frame
        int frameWidth = (int) capture.get(Videoio.CAP_PROP_FRAME_WIDTH);
        int frameHeight = (int) capture.get(Videoio.CAP_PROP_FRAME_HEIGHT);
        //System.out.println(frameHeight);
        //System.out.println(frameWidth);

        Rect leftRegion = new Rect(0,0,frameWidth/2, frameHeight);
        Rect rightRegion = new Rect(frameWidth/2,0,frameWidth/2,frameHeight);

        //create optical flow instances and divide the frame into half
        Mat prev_Frame = new Mat(), cur_Frame = new Mat();
        Mat previousGray = new Mat(), curGray = new Mat();
        float diff;


        //main loop
        capture.read(prev_Frame);
        while(true){
            capture.read(cur_Frame);
            if (cur_Frame.empty()) {
                break;
            }

            //preprocess frame
            Imgproc.cvtColor(cur_Frame,curGray,Imgproc.COLOR_BGR2GRAY);
            Imgproc.cvtColor(prev_Frame,previousGray,Imgproc.COLOR_BGR2GRAY);



            // Define keypoints in the left and right regions
            MatOfPoint previousLeftPts = new MatOfPoint();
            MatOfPoint previousRightPts = new MatOfPoint();
            MatOfPoint currentLeftPts = new MatOfPoint();
            MatOfPoint currentRightPts = new MatOfPoint();

            // Populate keypoints with corners from the left and right regions
            Imgproc.goodFeaturesToTrack(previousGray.submat(leftRegion), previousLeftPts, 100, 0.3, 7, new Mat(),7,false,0.04);
            Imgproc.goodFeaturesToTrack(previousGray.submat(rightRegion), previousRightPts, 100, 0.3, 7, new Mat(),7,false,0.04);

            //visualization
            Mat flow = new Mat();
            Video.calcOpticalFlowFarneback(previousGray,curGray,flow,0.2,5,10,5,5,1.2,0);
            Mat flowVisual = visualizeOpticalFlow(flow);

            HighGui.imshow("Original Left",cur_Frame.submat(leftRegion));
            HighGui.imshow("Original Right", cur_Frame.submat(rightRegion));
            HighGui.imshow("Flow",flowVisual);


            if (HighGui.waitKey(1)==27) {
                HighGui.destroyAllWindows();
                capture.release();
                System.exit(0);
                break;
            }

            // Convert keypoints to MatOfPoint2f
            MatOfPoint2f previousLeftPts2f = new MatOfPoint2f(previousLeftPts.toArray());
            MatOfPoint2f previousRightPts2f = new MatOfPoint2f(previousRightPts.toArray());


            //calculate optical flow
            MatOfPoint2f currentLeftPts2f = new MatOfPoint2f();
            MatOfPoint2f currentRightPts2f = new MatOfPoint2f();
            MatOfByte status = new MatOfByte();
            MatOfFloat err = new MatOfFloat();
            Video.calcOpticalFlowPyrLK(previousGray, curGray, previousLeftPts2f, currentLeftPts2f, status, err);
            Video.calcOpticalFlowPyrLK(previousGray, curGray, previousRightPts2f, currentRightPts2f, status, err);

            //calculate avg flow in left
            float left = sumFlow(previousLeftPts2f,currentLeftPts2f);
            left+=left*0.4;
            //calculate avg flow in right
            float right = sumFlow(previousRightPts2f,currentRightPts2f);




            //for debug
            System.out.println("left flow: "+left);
            System.out.println("right flow: "+right);

            if(left>right){
                System.out.println("turn right");
            }
            else if(left == right){
                System.out.println("no change");
            }
            else if (left<right){
                System.out.println("turn left");
            }

            //update prev_Frame
            prev_Frame = cur_Frame.clone();


        }


    }

    private static float sumFlow(MatOfPoint2f prev, MatOfPoint2f cur){
        float totalFlow = 0;

        Point[] prevArray = prev.toArray();
        Point[] curArray = cur.toArray();

        for(int i = 0; i < prevArray.length; i++){
            double dx = curArray[i].x - prevArray[i].x;
            double dy = curArray[i].y - prevArray[i].y;
            totalFlow += Math.sqrt(dx * dx + dy * dy);
        }

        return totalFlow;
    }

    private static Mat visualizeOpticalFlow(Mat flow) {
        ArrayList<Mat> flow_parts = new ArrayList<>(2);
        Core.split(flow, flow_parts);
        Mat magnitude = new Mat(), angle = new Mat(), magn_norm = new Mat();
        Core.cartToPolar(flow_parts.get(0), flow_parts.get(1), magnitude, angle,true);
        Core.normalize(magnitude, magn_norm,0.0,1.0, Core.NORM_MINMAX);
        float factor = (float) ((1.0/360.0)*(180.0/255.0));
        Mat new_angle = new Mat();
        Core.multiply(angle, new Scalar(factor), new_angle);
        //build hsv image
        ArrayList<Mat> _hsv = new ArrayList<>() ;
        Mat hsv = new Mat(), hsv8 = new Mat(), bgr = new Mat();
        _hsv.add(new_angle);
        _hsv.add(Mat.ones(angle.size(), CvType.CV_32F));
        _hsv.add(magn_norm);
        Core.merge(_hsv, hsv);
        hsv.convertTo(hsv8, CvType.CV_8U, 255.0);
        Imgproc.cvtColor(hsv8, bgr, Imgproc.COLOR_HSV2BGR);
        return bgr;
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new ObjectAvoidance().cam_OptFlow(null,false,0);
    }
}
