import ai.onnxruntime.OrtException;
import crnn.Recognition;
import crnn.RegModel;
import dbnet.DetModel;
import dbnet.Detection;
import object.Box;
import object.Boxes;
import object.Text;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.List;

/**
 * @author sy
 * @date 2024/7/31 21:47
 */
public class DetRecTest {
    public static void main(String...args) throws OrtException {
        nu.pattern.OpenCV.loadLocally();

        // init and load dbnet
        String detModelPath = "models\\det\\inference.onnx";

        DetModel detModel = new DetModel(detModelPath);
        Detection detection = new Detection(detModel);

        // init and load crnn
        String recogModelPath = "models\\rec\\inference.onnx";
        String vocabPath = "models\\rec\\vocab.txt";

        RegModel regModel = new RegModel(recogModelPath, vocabPath);
        Recognition recognition = new Recognition(regModel);

        // test
        String imgPath = "imgs/det/test1.jpg";
        Mat img = Imgcodecs.imread(imgPath);

        double startTime = System.currentTimeMillis()/1000.0;

        Boxes detBoxes = detection.detect(img);
        List<Box> boxList = detBoxes.getBoxes();
        int size = boxList.size();
        for(int i=size-1; i>=0; i--) {
            float[] linePosition = boxList.get(i).getLinePosition();
            int x0 = (int) linePosition[0];
            int y0 = (int) linePosition[1];
            int x2 = (int) linePosition[2];
            int y2 = (int) linePosition[3];
            Mat subMat = img.submat(y0, y2, x0, x2);
            Text text = recognition.recognize(subMat);
            System.out.println(text);
            subMat.release();
        }

        System.out.println("时间： " + (System.currentTimeMillis()/1000.0 - startTime));

        img.release();
    }
}
