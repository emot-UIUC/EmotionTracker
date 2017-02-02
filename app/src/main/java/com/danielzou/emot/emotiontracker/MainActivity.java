package com.danielzou.emot.emotiontracker;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Context;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.PowerManager;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.JavaCameraView;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class MainActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getName();

    // Loads camera view of OpenCV for us to use. This lets us see using OpenCV
    private CameraBridgeViewBase mOpenCvCameraView;

    private ProgressDialog mProgressDialog;

    // These variables are used (at the moment) to fix camera orientation from 270degree to 0degree
    Mat mRgba;
    Mat mRgbaF;
    Mat mRgbaT;
    Mat mGray;
    private CascadeClassifier faceCascade;
    private File mCascadeFile;
    EmotionRecognizer emotionRecognizer;

    public MainActivity() {
        System.loadLibrary("opencv_java3");

        File downloadsFolderPath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        String downloadFolderPathString = downloadsFolderPath + "/";
        Log.e(TAG, "Directory for downloads: " + downloadFolderPathString);
        emotionRecognizer = new EmotionRecognizer(downloadFolderPathString); //Downloads folder

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(com.danielzou.emot.emotiontracker.R.layout.activity_main);

        mOpenCvCameraView = (JavaCameraView) findViewById(com.danielzou.emot.emotiontracker.R.id.java_surface_view);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

        //Instantiate progress dialog in onCreate
        mProgressDialog = new ProgressDialog(MainActivity.this);
        mProgressDialog.setMessage("Downloading training data...");
        mProgressDialog.setIndeterminate(true);
        mProgressDialog.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
        mProgressDialog.setCancelable(false);

        //Download training set for emotion classifier
//        final DownloadTask downloadTask = new DownloadTask(MainActivity.this);
//        downloadTask.execute("https://s3.amazonaws.com/bretl-emot/dataset/anger/0.jpg");
    }


    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
        mRgbaT = new Mat(width, width, CvType.CV_8UC4);
        mGray = new Mat(width, width, CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    private void setText(final TextView text, final String value){
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                text.setText(value);
            }
        });
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        //Rotate mRgba 90 degrees
//        Core.transpose(mRgba, mRgbaT);
//        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 0,0, 0);
//        Core.flip(mRgbaF, mRgba, 1 );

        //Convert image to grayscale to improve detection speed and accuracy
        Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_RGBA2GRAY);
        MatOfRect faceDetections = new MatOfRect();
        faceCascade.detectMultiScale(mGray, faceDetections);
        Log.i(TAG, String.format("Detected %s faces", faceDetections.toArray().length));
        for (Rect rect : faceDetections.toArray()) {
            Imgproc.rectangle(mGray, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0), 2);
            String emotion = emotionRecognizer.emotionsArr[emotionRecognizer.detectEmotion(mGray.submat(rect))];
            //Imgproc.putText(mGray, "Detected " + emotion, new Point(0,0), Core.FONT_HERSHEY_TRIPLEX, 2.0, new  Scalar(0,255,255));
//            runOnUiThread(new Runnable() {
//                @Override
//                public void run() {
//
//                //stuff that updates ui
//                    TextView textView = (TextView) findViewById(R.id.textview);
//                    textView.setText("Detected" + emotion);
//                }
//            });
            setText((TextView)findViewById(R.id.textview), "Detected: " + emotion);

            Log.i(TAG, String.format("Detected " + emotion));
        }

        if (faceDetections.toArray().length == 1) {

        } else {
            Log.i(TAG, String.format("No/multiple faces detected, passing over frame"));
        }

        return mGray; // This function must return
    }

    /**
     * Crop the given face
     * @param gray
     * @param face
     * @return
     */
    public Mat cropFace(Mat gray, Rect face) {
        Mat faceSlice = gray.submat(face);
        return faceSlice;
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();

                    try {
                        InputStream is = getApplicationContext().getResources().openRawResource(com.danielzou.emot.emotiontracker.R.raw.haarcascade_frontalface_default);
                        FileOutputStream os = openFileOutput("haarcascade_frontalface_default.xml", MODE_PRIVATE);
                        mCascadeFile = getFileStreamPath("haarcascade_frontalface_default.xml");

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    String faceCascadeName = mCascadeFile.getAbsolutePath();
                    faceCascade = new CascadeClassifier(faceCascadeName);
                    if(faceCascade.empty()) {
                        System.out.println("--(!)Error loading A\n");
                        return;
                    }
                    else {
                        System.out.println("Face classifier loaded up");
                    }
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };



    /**
     * Thanks to:
     * http://stackoverflow.com/questions/3028306/download-a-file-with-android-and-showing-the-progress-in-a-progressdialog
     */
    public class DownloadTask extends AsyncTask<String, Integer, String> {

        private Context context;
        private PowerManager.WakeLock mWakeLock;

        public DownloadTask(Context context) {
            this.context = context;
        }

        @Override
        protected String doInBackground(String... sUrl) {
            InputStream input = null;
            OutputStream output = null;
            HttpURLConnection connection = null;

            try {
                URL url = new URL(sUrl[0]);
                connection = (HttpURLConnection) url.openConnection();
                connection.connect();

                // Expect HTTP 200 OK
                if (connection.getResponseCode() != HttpURLConnection.HTTP_OK) {
                    return "Server returned HTTP " + connection .getResponseCode()
                            + " " + connection.getResponseMessage();
                }

                // For displaying download percentage; -1 if length not reported by the server
                int fileLength = connection.getContentLength();

                // Download
                input = connection.getInputStream();
                String fileName = "anger-0";
                output = new FileOutputStream("/sdcard/" + fileName);

                byte data[] = new byte[4096];
                long total = 0;
                int count;
                while ((count = input.read(data)) != 1) {
                    total += count;
                    // Publishing the progress
                    if (fileLength > 0) // Only if the length is known
                        publishProgress((int) (total * 100 / fileLength));
                    output.write(data, 0, count);
                }
            } catch (Exception e) {
                return e.toString();
            } finally {
                try {
                    if (output != null)
                        output.close();
                    if (input != null)
                        input.close();
                } catch (IOException ignored) {
                }

                if (connection != null)
                    connection.disconnect();
            }
            return null;
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            // Take CPU lock to prevent CPU from going off if user presses power button
            PowerManager pm = (PowerManager) context.getSystemService(Context.POWER_SERVICE);
            mWakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, getClass().getName());
            mWakeLock.acquire();
            mProgressDialog.show();
        }

        @Override
        protected void onProgressUpdate(Integer... progress) {
            super.onProgressUpdate(progress);
            // If we get here, the length is known, now set indeterminate to false
            mProgressDialog.setIndeterminate(false);
            mProgressDialog.setMax(100);
            mProgressDialog.setProgress(progress[0]);
        }

        @Override
        protected void onPostExecute(String result) {
            mWakeLock.release();
            mProgressDialog.dismiss();
            if (result != null) {
                Toast.makeText(context, "Download error: " + result, Toast.LENGTH_LONG).show();
            } else {
                Toast.makeText(context, "File downloaded", Toast.LENGTH_SHORT).show();
            }
        }
    }

}

