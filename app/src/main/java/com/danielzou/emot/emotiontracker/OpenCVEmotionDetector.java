package com.danielzou.emot.emotiontracker;

import android.util.Log;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.bytedeco.javacpp.opencv_face.BasicFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;

/**
 * Created using sample from Petter Christian Bjelland at
 * http://pcbje.com/2012/12/doing-face-recognition-with-javacv/.
 * As well as instructions from Paul Vangent at
 * http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/.
 */

public class OpenCVEmotionDetector {

    private static final String TAG = OpenCVEmotionDetector.class.getName();

    BasicFaceRecognizer mFaceRecognizer = createFisherFaceRecognizer();
    final String[] emotionsArr = {"neutral", "anger", "contempt", "disgust", "fear", "happy"};
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
            }
        }

        Log.e(TAG, "Training fisher face classifier.");
        Log.e(TAG, "Size of training set is: " + trainingLabels.size() + " images");
        Mat trainingDataArr[] = new Mat[trainingData.size()];
        for (int i = 0; i < trainingData.size(); i++) {
            Log.e(TAG, "Moved training data matrix: " + trainingData.get(i));
            trainingDataArr[i] = trainingData.get(i);
        }
        opencv_core.MatVector matVectorOfTrainingData = new MatVector(trainingDataArr);
        int[] trainingLabelsArr = new int[trainingLabels.size()];
        for (int i = 0; i < trainingLabels.size(); i++) {
            Log.e(TAG, "Moved training labels: " + trainingLabels.get(i));
            trainingLabelsArr[i] = trainingLabels.get(i);
        }
        opencv_core.Mat matOfTrainingLabels = new Mat(trainingLabelsArr);
        mFaceRecognizer.train(matVectorOfTrainingData, matOfTrainingLabels);

        Log.e(TAG, "Predicting classification set");
        int count = 0;
        int correct = 0;
        int incorrect = 0;
        for (Mat image : predictionData) {
            int prediction = detectEmotion(image);
            if (prediction == predictionLabels.get(count)) {
                correct++;
                count++;
            } else {
                incorrect++;
                count++;
            }
        }
        Log.i(TAG, "Classifier accuracy is: " + ((100 * correct)/(correct + incorrect)));
    }

    /**
     * Splits data set in training set and prediction set 80-20.
     * @param trainingDir
     * @param emotion
     * @return Returns the training set in the 0 index and the prediction set in the 1 index.
     */
    public List<List<File>> getFiles(String trainingDir, String emotion) {
        String emotionDirectoryPath = trainingDir + "/dataset/" + emotion + "/"; // File path on mac uses forward slashes. Change if on Windows or Linux.
        File emotionDir = new File(emotionDirectoryPath);
        List<File> files = Arrays.asList(emotionDir.listFiles());
        Collections.shuffle(files);
        Log.e(TAG, "Training set size: " + (int)(.8 * files.size()));
        List<File> training = files.subList(0, (int)(.8 * files.size()));
        List<File> prediction = files.subList((int)(.8 * files.size()) + 1, files.size());
        List<List<File>> sets = new ArrayList<>(2);
        sets.add(0, training);
        sets.add(1, prediction);
        return sets;
    }

    public int detectEmotion(Mat image) {
        int predictedLabel = mFaceRecognizer.predict(image);

        return predictedLabel;
    }
}
