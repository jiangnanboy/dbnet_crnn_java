import ai.onnxruntime.OrtException;
import object.Boxes;
import dbnet.DetModel;
import dbnet.Detection;
import utils.common.ImageUtils;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * @author sy
 * @date 2024/7/1 21:26
 */
public class DBNetTest {
    public static void main(String...args) throws OrtException {
        nu.pattern.OpenCV.loadLocally();
        String imgPath = "imgs/det/test4.jpg";
        Mat img = Imgcodecs.imread(imgPath);
        String modelPath = "models\\det\\inference.onnx";
        DetModel detModel = new DetModel(modelPath);
        Detection detection = new Detection(detModel);
        double startTime = System.currentTimeMillis()/1000.0;
        Boxes detBoxes = detection.detect(img);
        System.out.println("时间： " + (System.currentTimeMillis()/1000.0 - startTime));
        ImageUtils.drawPredictions(img, detBoxes);
        Imgcodecs.imwrite("imgs\\prediction.jpg", img);
        img.release();
    }
}
