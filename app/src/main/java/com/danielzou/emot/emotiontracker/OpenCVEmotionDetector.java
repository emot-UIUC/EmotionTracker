package com.danielzou.emot.emotiontracker;

import android.util.Log;

import java.io.File;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_imgproc;

/**
 * Created using sample from Petter Christian Bjelland at
 * http://pcbje.com/2012/12/doing-face-recognition-with-javacv/.
 * As well as instructions from Paul Vangent at
 * http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/.
 */

public class OpenCVEmotionDetector {

    private static final String TAG = OpenCVEmotionDetector.class.getName();

    FaceRecognizer mFaceRecognizer = createFisherFaceRecognizer();
    //final String[] emotionsArr = {"neutral", "anger", "contempt", "disgust", "fear", "happy"};
    final String[] emotionsArr = {"anger", "contempt", "disgust", "fear", "happy"};
    final List<String> emotions = Arrays.asList(emotionsArr);
    List<Mat> trainingData = new ArrayList<>();
    List<Integer> trainingLabels = new ArrayList<>();
    List<Mat> predictionData = new ArrayList<>();
    List<Integer> predictionLabels = new ArrayList<>();

    public OpenCVEmotionDetector(String trainingDirectoryPath) {
        String trainingDir = trainingDirectoryPath;
        for (String emotion : emotions) {
            List<File> training = getFiles(trainingDir, emotion).get(0);
            List<File> prediction = getFiles(trainingDir, emotion).get(1);
            for (File item : training) {
                Mat gray = imread(item.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
                trainingData.add(gray);
                trainingLabels.add(emotions.indexOf(emotion));
            }
            for (File item : prediction) {
                Mat gray = imread(item.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
                predictionData.add(gray);
                predictionLabels.add(emotions.indexOf(emotion));
                //Log.e(TAG, "Prediction Label: " + emotions.indexOf(emotion));
            }
        }

        Log.e(TAG, "Training fisher face classifier.");
        Log.e(TAG, "******Size of training data set (data) is: " + trainingData.size() + " images");
        Log.e(TAG, "******Size of training data set (labels) is: " + trainingLabels.size() + " images");
        for (int label:
             trainingLabels) {
            Log.e(TAG, emotionsArr[label]);
        }
        /**
         * Developed with the help of the JavaCV OpenCVFaceRecognizer sample
         */
        //trainingData = trainingData.subList(309,trainingData.size());
        //trainingLabels = trainingLabels.subList(309,trainingLabels.size());
        opencv_core.MatVector matVectorOfTrainingData = new MatVector(trainingData.size());
        opencv_core.Mat matOfTrainingLabels = new Mat(trainingLabels.size(), 1, CV_32SC1);
        IntBuffer labelsBuf = matOfTrainingLabels.createBuffer();
        int counter = 0;
        for (Mat image : trainingData) {
            matVectorOfTrainingData.put(counter, image);
            counter++;
        }
        counter = 0;
        for (int label : trainingLabels) {
            labelsBuf.put(counter, label);
            counter++;
        }
//        for (int i = 0; i < matVectorOfTrainingData.size(); i++) {
//            Log.e(TAG, "" + i + matVectorOfTrainingData.get(i).toString());
//        }
        //Log.e(TAG, "Mat of training labels: " + matOfTrainingLabels.toString());

        mFaceRecognizer.train(matVectorOfTrainingData, matOfTrainingLabels);
        //Log.e(TAG, "Predicting classification set");
//        int count = 0;
//        int correct = 0;
//        int incorrect = 0;
//        for (Mat image : predictionData) {
//            int prediction = detectEmotion(image);
//            if (prediction == predictionLabels.get(count)) {
//                correct++;
//                count++;
//            } else {
//                incorrect++;
//                count++;
//            }
//        }
//        Log.i(TAG, "Classifier accuracy is: " + ((100 * correct)/(correct + incorrect)));
    }

    /**
     * Splits data set in training set and prediction set 80-20.
     * @param trainingDir
     * @param emotion
     * @return Returns the training set in the 0 index and the prediction set in the 1 index.
     */
    public List<List<File>> getFiles(String trainingDir, String emotion) {
        String emotionDirectoryPath = trainingDir + "dataset/" + emotion + "/"; // File path on mac uses forward slashes. Change if on Windows or Linux.
        Log.e(TAG, "Root directory: " + emotionDirectoryPath);
        File emotionDir = new File(emotionDirectoryPath);
        List<File> files = Arrays.asList(emotionDir.listFiles());
//        for(File file : files) {
//            Log.e(TAG, "File: " + file.getName());
//        }
        //Collections.shuffle(files);

        //Log.e(TAG, "Training set size: " + (int)(.8 * files.size()));
        //List<File> training = files.subList(0, (int)(.8 * files.size()));

        List<File> training = files;
        Log.e(TAG, "Training set size: " + training.size());
        List<File> prediction = files.subList((int)(.8 * files.size()) + 1, files.size());
        Log.e(TAG, "Prediction set size: " + prediction.size());
        List<List<File>> sets = new ArrayList<>();
        sets.add(0, training);
        sets.add(1, prediction);
        return sets;
    }

    /**
     * Conversion from opencv to JavaCV mat https://github.com/bytedeco/javacpp/issues/38
     * @param image
     * @return
     */
    public int detectEmotion(org.opencv.core.Mat image) {
        final org.opencv.core.Mat image2 = image;
        Mat imageJavaCV = new Mat((Pointer)null) { { address = image2.getNativeObjAddr(); } };
        /**
         * Reshape needs clone - http://answers.opencv.org/question/22042/reshape-function-problem/
            */
        Mat imageJavaCVResized = new Mat();
        opencv_imgproc.resize(imageJavaCV, imageJavaCVResized, new opencv_core.Size(350,350));

        int predictedLabel = mFaceRecognizer.predict(imageJavaCVResized);

        return predictedLabel;
    }

    public int detectEmotion(Mat image) {
        int predictedLabel = mFaceRecognizer.predict(image);

        return predictedLabel;
    }
}
