package org.diploma.diplomaproject;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import static java.lang.Math.exp;
import org.pytorch.Device;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
//import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import android.util.Log;
import android.os.SystemClock;
import java.util.Map;
import java.util.HashMap;
import static java.util.Arrays.asList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    /*
    Utils
     */
    // class2color
    Map<Integer, Integer> class2color = new HashMap<Integer, Integer>() {{
        put(0, 0xFF000000); // background
        put(1, 0xFFFF0000); // lips
        put(2, 0xFF00FF00); // eye
        put(3, 0xFF0000FF); // nose
        put(4, 0xFFFFFF00); // hair
        put(5, 0xFFFF00FF); // eyebrows
        put(6, 0xFFFFFFFF); // teeth
        put(7, 0xFF808080); // face
        put(8, 0xFF00FFFF); // ears
        put(9, 0xFF008080); // glasses
        put(10, 0xFFFFc0c0); // beard
    }};
    // class order
    List<Integer> class_order = asList(0, 7, 4, 2, 5, 6, 3, 1, 8, 10, 9);

    private static int RESULT_LOAD_IMAGE = 1;


    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button buttonLoadImage = (Button) findViewById(R.id.button);
        Button detectButton = (Button) findViewById(R.id.detect);


        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }
        // Load Image
        buttonLoadImage.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View arg0) {
                TextView textView = findViewById(R.id.result_text);
                textView.setText("");
                Intent i = new Intent(
                        Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

                startActivityForResult(i, RESULT_LOAD_IMAGE);


            }
        });

        // Segmentation
        detectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                // Init Bitmap and module
                Bitmap mBitmap = null;
                Module module = null;

                //Getting the image from the image view
                ImageView mImageView = findViewById(R.id.image);
                mBitmap = ((BitmapDrawable)mImageView.getDrawable()).getBitmap();

                int initial_width = mBitmap.getWidth();
                int initial_height = mBitmap.getHeight();

                mBitmap = Bitmap.createScaledBitmap(mBitmap, 320, 320, true); // convert to 320x320
                mImageView.setImageBitmap(mBitmap);

                // read model
                try {
                    if (module == null) {
//                        module = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "uu.ptl"));
                        module = Module.load(MainActivity.assetFilePath(getApplicationContext(), "uu.pt"));
                    }
                } catch (IOException e) {
                    Log.e("Image Segmentation", "Error reading assets", e);
                    finish();
                }

                // normalize image
                float[] zeros = new float[]{0f, 0f, 0f};
                float[] ones = new float[]{1f, 1f, 1f};
                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap, zeros, ones);
                final float[] inputs = inputTensor.getDataAsFloatArray();

                // forward
                final long startTime = SystemClock.elapsedRealtime();
                final Tensor outTensor = module.forward(IValue.from(inputTensor)).toTensor();
//                Map<String, IValue> outputTensor = module.forward(IValue.from(inputTensor)).toDictStringKey();
                final long inferenceTime = SystemClock.elapsedRealtime() - startTime;


                // get list of values
//                final Tensor outTensor = outputTensor.get("out").toTensor();
                final float[] scores = outTensor.getDataAsFloatArray();

                // get image shape
                int width = mBitmap.getWidth();
                int height = mBitmap.getHeight();
                // init image mask
                int[] intValues = new int[width * height];

                // show inference time
                TextView textView = findViewById(R.id.result_text);
                textView.setText("Inference time, ms: " + String.valueOf(inferenceTime));
//                textView.setText("Inference time, ms" + String.valueOf(scores[42]) + "_" + (float) (1.0 / (1.0 + exp(-scores[42]))));

                // draw classes
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        // default color is black
                        intValues[h * width + w] = 0xFF000000;
                        // for each value in class order
                        double best_score = 0.0;
                        int filled_class = 0;
                        for (Integer class_val : class_order) {
                            // get score
                            float score = scores[class_val * (width * height) + h * width + w];
                            // sigmoid
                            score = (float) (1.0 / (1.0 + exp(-score)));
                            // if score >= best_score
                            if (score >= best_score) {
                                best_score = score;
                                filled_class = class_val;

                            }
                        }
                        intValues[h * width + w] = class2color.get(filled_class);
                    }
                }

                // draw results
                Bitmap bmpSegmentation = Bitmap.createScaledBitmap(mBitmap, width, height, true);
                Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
                outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
//                final Bitmap transferredBitmap = Bitmap.createScaledBitmap(outputBitmap, mBitmap.getWidth(), mBitmap.getHeight(), false);
                final Bitmap transferredBitmap = Bitmap.createScaledBitmap(outputBitmap, initial_width, initial_height, false);
                mImageView.setImageBitmap(transferredBitmap);

            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //This functions return the selected image from gallery
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };

            Cursor cursor = getContentResolver().query(selectedImage,
                    filePathColumn, null, null, null);
            cursor.moveToFirst();

            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();

            ImageView imageView = (ImageView) findViewById(R.id.image);
            imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));

            //Setting the URI so we can read the Bitmap from the image
            imageView.setImageURI(null);
            imageView.setImageURI(selectedImage);


        }


    }

}