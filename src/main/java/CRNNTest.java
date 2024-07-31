import ai.onnxruntime.OrtException;
import object.Text;
import crnn.Recognition;
import crnn.RegModel;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * @author sy
 * @date 2024/7/5 21:31
 */
public class CRNNTest {
    public static void main(String...args) throws OrtException {
        nu.pattern.OpenCV.loadLocally();
        String imgPath = "imgs/recog/test12.png";
        Mat img = Imgcodecs.imread(imgPath);
        String modelPath = "models\\rec\\inference.onnx";
        String vocabPath = "models\\rec\\vocab.txt";
        RegModel regModel = new RegModel(modelPath, vocabPath);
        Recognition recognition = new Recognition(regModel);

        double startTime = System.currentTimeMillis()/1000.0;

        Text text = recognition.recognize(img);
        System.out.println(text);
        System.out.println("时间： " + (System.currentTimeMillis()/1000.0 - startTime));

        img.release();
    }
}
