import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.util.ArrayList;



public class ObjectAvoidance {
    public void cam_OptFlow(int channel, boolean visualization, double biasL, double biasR){
        VideoCapture capture;
        //capture video frames
        //System.out.println("use "+channel+" channel");
        capture = new VideoCapture(channel);

        if (!capture.isOpened()) {
            System.out.println("Unable to capture");
            System.exit(-1);
        }

        //create Video Frame
        int frameWidth = (int) capture.get(Videoio.CAP_PROP_FRAME_WIDTH);
        int frameHeight = (int) capture.get(Videoio.CAP_PROP_FRAME_HEIGHT);
        //System.out.println(frameHeight); //480
        //System.out.println(frameWidth); //640

        Rect leftRegion = new Rect(0,0,frameWidth/2, frameHeight);
        Rect rightRegion = new Rect(frameWidth/2,0,frameWidth/2,frameHeight);

        //Rect leftRegion = new Rect(100,50,220, 430);
        //Rect rightRegion = new Rect(frameWidth/2,50,220,430);

        //create optical flow instances and divide the frame into half
        Mat prev_Frame = new Mat(), cur_Frame = new Mat();
        Mat previousGray = new Mat(), curGray = new Mat();

        double thresholdLeft = Double.MIN_VALUE, thresholdRight = Double.MIN_VALUE;
        double tmp = 0, tmp2 = 0;

        //Jiahao: Create a background subtractor
        BackgroundSubtractorMOG2 subtractor = Video.createBackgroundSubtractorMOG2();




        //main loop
        int maxStep = 40;
        int i = 0; //for configuration
        int j = 0;

        capture.read(prev_Frame);
        while(true){
            capture.read(cur_Frame);
            if (cur_Frame.empty()) {
                break;
            }

            //preprocess frame
            Imgproc.cvtColor(cur_Frame,curGray,Imgproc.COLOR_BGR2GRAY);
            Imgproc.cvtColor(prev_Frame,previousGray,Imgproc.COLOR_BGR2GRAY);

            //Jiahao: apply background subtraction...
            Mat fgMask = new Mat();
            subtractor.apply(curGray, fgMask);

            //Jiahao: Now fgMask contains the foreground mask. We can use this to mask out
            // the background when calculating the optical flow.
            Mat maskedGray = new Mat();
            curGray.copyTo(maskedGray, fgMask);



            if(visualization){
                HighGui.imshow("original left",cur_Frame.submat(leftRegion));
                HighGui.imshow("original right",cur_Frame.submat(rightRegion));

                if(HighGui.waitKey(1)==27){
                    capture.release();
                    HighGui.destroyAllWindows();
                    System.exit(-1);
                }
            }

            //prepare for calculation
            MatOfPoint prevLeftPoints = new MatOfPoint(), prevRightPoints = new MatOfPoint();
            MatOfPoint curLeftPoints = new MatOfPoint(), curRightPoints = new MatOfPoint();
            //find good features to track
            Imgproc.goodFeaturesToTrack(previousGray.submat(leftRegion),prevLeftPoints,600,0.7,14, new Mat(),7,false,0.04);
            Imgproc.goodFeaturesToTrack(previousGray.submat(rightRegion),prevRightPoints,560,0.8,10, new Mat(),7,false,0.04);
            //calculate optical flow
            MatOfPoint2f prevL2f = new MatOfPoint2f(prevLeftPoints.toArray()), prevR2f = new MatOfPoint2f(prevRightPoints.toArray());
            MatOfPoint2f resultL = new MatOfPoint2f(),resultR = new MatOfPoint2f();
            //MatOfPoint2f curL2f = new MatOfPoint2f(curLeftPoints.toArray()), curR2F = new MatOfPoint2f(curRightPoints.toArray());
            MatOfByte status = new MatOfByte();
            MatOfFloat err = new MatOfFloat();
            TermCriteria criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,10,0.03);

            if(prevL2f.empty() || prevR2f.empty()){
                System.out.println("Not enough feature points. Waiting for next collection of data");

            }
            else {

                Video.calcOpticalFlowPyrLK(previousGray, curGray, prevL2f, resultL, status, err, new Size(15, 15), 2, criteria);
                Video.calcOpticalFlowPyrLK(previousGray, curGray, prevR2f, resultR, status, err, new Size(15, 15), 2, criteria);

                //convert the result to an array of vectors
                Point left_arr[] = resultL.toArray();
                Point right_arr[] = resultR.toArray();


                if (i > maxStep) {
                    //find out direction to move

                    //Conditions:
                    //1. left max <= left threshold and right max <= right threshold: both sides have decrease in objects
                    //  Thus, forward
                    //2. left max > left threshold and right max <= right threshold: left side have increase in objects
                    //  Thus, turn right
                    //3. left max <= left threshold and right max > right threshold: right side have increase in objects
                    //  Thus, turn left
                    //4. left max > left threshold and right max > right threshold: run into an obstcle
                    //  Thus, stop

                    //Unable to do:
                    //6. Two Infer-red sensor output the same distance and it is too close to an object: run into a wall
                    //  Stop

                    //if the max - threshold is within 10% of the threshold, then it would be considered as the same
                    //if max - threshold is greater than 10% of the threshold:
                    //  a. max - threshold < 0: smaller
                    //  b. max - threshold > 0: greater

                    //find max vector in left and right half
                    double maxLeft = findMaxMagnitude(left_arr);
                    double maxRight = findMaxMagnitude(right_arr);

                    //for debug////////////////////////////////////
                    //System.out.println("-----------------------DATA----------------------");
                    //System.out.println("Max left flow: " + maxLeft);
                    //System.out.println("Max right flow: " + maxRight);
                    //System.out.println("Left Threshold: " + thresholdLeft);
                    //System.out.println("Right Threshold: " + thresholdRight);
                    //System.out.println("-------------------------------------------------");
                    /////////////////////////////////////////////

                    double diffLeft = maxLeft - thresholdLeft, diffRight = maxRight - thresholdRight;
                    double absLeft = Math.abs(diffLeft), absRight = Math.abs(diffRight);
                    double limitLeft = 0.07 * thresholdLeft, limitRight = 0.05 * thresholdRight;
                    String cmd = null;

                    //System.out.println("diff left: " + diffLeft);
                    //System.out.println("diff right: " + diffRight);
                    //System.out.println();



                    if(absLeft <= limitLeft && absRight <= limitRight){
                        //condition 1
                        cmd="forward";

                    }
                    else if(absLeft > limitLeft && absRight <= limitRight){
                        //condition 2
                        cmd = "right";

                    }
                    else if(absLeft <= limitLeft && absRight > limitRight){
                        //condition 3
                        cmd = "left";
                    }
                    else if(absLeft > limitLeft && absRight > limitRight){
                        //condition 4
                        cmd = "stop";
                    }



                    if(cmd!=null) {
                        System.out.println("-------------------- Command Output -------------------");
                        System.out.println(cmd);
                        //j++;
                        System.out.println("-------------------------------------------------------");

                    }

                    //recaliberate
                    /*if(j>=100){
                        System.out.println("======================Start Recaliberating===================");
                        j = 0;
                        i = 0;
                    }*/

                } else {
                    //when i is smaller and equal to 15 it will halt and initialize the thresholds based on
                    //the current environment

                    //find the threshold

                    tmp += findMaxMagnitude(left_arr);
                    tmp2 += findMaxMagnitude(right_arr);
                    System.out.println("----------------------- Halt. Initializing threshold. Step " + i + " ------------");

                    if (i == maxStep) {
                        thresholdLeft = tmp / (i + 1)+biasL;
                        thresholdRight = tmp2 / (i + 1)+biasR;
                        System.out.println("------------------ FINISH INITIALIZATION ---------------------");
                        //System.out.println("Left Final Threshold: " + thresholdLeft);
                        //System.out.println("Right Final Threshold: " + thresholdRight);
                        System.out.println("==============================================================");
                    }


                    i++;

                }
            }


            //update prev_Frame
            prev_Frame = cur_Frame.clone();


        }


    }



    private static double findMaxMagnitude(Point[] vector){
        double maxMag = Double.MIN_VALUE;

        for(int i = 0; i < vector.length; i++){

            //get the flow vector
            Point flowVector = vector[i];
            double magnitude = calculateMagnitude(flowVector);

            if(magnitude>maxMag ){
                //find the vector that is moving towards the robot with maximum velocity
                maxMag = magnitude;
            }

        }
        //System.out.println("Max mag among this data set: "+maxMag);
        return maxMag;
    }


    private static double calculateMagnitude(Point vector){

        return Math.sqrt(vector.x*vector.x+vector.y*vector.y);
    }

    private static Mat visualizeOpticalFlow_Farneback(Mat flow) {
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

    private static Mat visualizeOpticalFlow_PyrLK(){
        return null;
    }
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new ObjectAvoidance().cam_OptFlow(0,true,0,0);
    }
}
