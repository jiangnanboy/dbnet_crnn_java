package crnn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import object.Text;
import utils.common.CollectionUtil;
import utils.common.ImageUtils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.Map;

/**
 * @author sy
 * @date 2024/7/5 21:28
 */
public class Recognition {
    RegModel regModel = null;
    public Recognition(RegModel regModel) {
        this.regModel = regModel;
    }

    /**
     * @param imgMat
     * @return
     * @throws OrtException
     */
    public Text recognize(Mat imgMat) throws OrtException {
        Map<String, OnnxTensor> inputMap = this.prepareInput(imgMat);
        float[][] predictions = inference(inputMap);
        Text text = this.processOutput(predictions);
        return text;
    }

    /**
     * @param img
     * @return
     * @throws OrtException
     */
    private Map<String, OnnxTensor> prepareInput(Mat img) throws OrtException {
        int rawImgHeight = img.height();
        int rawImgWidth = img.width();
        float curRatio = rawImgWidth / (float)rawImgHeight;
        int maskHeight = 32;
        int maskWidth = 640;
        int curTargetHeight;
        int curTargetWidth;
        if(curRatio > (maskWidth / (float)maskHeight)) {
            curTargetHeight = maskHeight;
            curTargetWidth = maskWidth;
        } else {
            curTargetHeight = maskHeight;
            curTargetWidth = (int) (maskHeight * curRatio);
        }

        Mat inputImg = new Mat();
        Imgproc.resize(img, inputImg, new Size(curTargetWidth, curTargetHeight));
        Mat maskMat = Mat.zeros(maskHeight, maskWidth, CvType.CV_8UC3);

        //创建一行像素点的缓存数组
        byte[] data = new byte[inputImg.channels() * inputImg.width()];
        for(int row = 0; row < inputImg.height(); row++){
            inputImg.get(row,0, data);//当col为0的时候表示获取一行
            //写入
            maskMat.put(row,0, data);
        }
        inputImg = maskMat;

        inputImg.convertTo(inputImg, CvType.CV_32FC3, 1./255);

        Map<String, OnnxTensor> inputMap = CollectionUtil.newHashMap();
        float[] whc = new float[(int) (maskHeight * maskWidth * 3)];
        inputImg.get(0, 0, whc);
        float[] chw = ImageUtils.whc2cwh(whc);

        FloatBuffer inputBuffer= FloatBuffer.wrap(chw);
        OnnxTensor inputTensor = OnnxTensor.createTensor(this.regModel.getEnv(), inputBuffer, new long[]{1, 3, maskHeight, maskWidth});

        inputMap.put(this.regModel.getInputName(), inputTensor);
        maskMat.release();
        inputImg.release();
        return inputMap;
    }

    /**
     * @param inputMap
     * @return
     * @throws OrtException
     */
    private float[][] inference(Map<String, OnnxTensor> inputMap) throws OrtException {
        OrtSession.Result result = this.regModel.getSession().run(inputMap);
        float[][] predicitons = ((float[][][])result.get(0).getValue())[0];
        return predicitons;
    }

    /**
     * @param predictions
     * @return
     */
    private Text processOutput(float[][] predictions) {
        NDManager ndManager = NDManager.newBaseManager();
        NDArray ndArray = ndManager.create(predictions);

        NDArray ndArraySoftmax = ndArray.softmax(-1);
        NDArray ndArrayArgmax = ndArraySoftmax.argMax(-1);

        long lastP = 0L;
        StringBuilder sb = new StringBuilder();
        for(int i=0; i<ndArrayArgmax.size(); i++) {
            long index = ndArrayArgmax.getLong(i);
            if((index != lastP) && (index != 0L)) {
                sb.append(this.regModel.getLabelMapping().get(index));
            }
            lastP = index;
        }
        Text text = new Text(sb.toString());
        return text;
    }

    /**
     * @param ndArray
     * @return
     */
    public NDArray softmax(NDArray ndArray) {
        NDArray ndArrayExp = ndArray.exp();
        NDArray partition = ndArrayExp.sum(new int[]{1}, true);
        return ndArrayExp.div(partition); // 这里应用了广播机制
    }

}
