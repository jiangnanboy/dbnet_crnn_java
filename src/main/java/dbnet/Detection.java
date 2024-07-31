package dbnet;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import object.Box;
import object.Boxes;
import utils.common.CollectionUtil;
import utils.common.ImageUtils;
import utils.cv.NDArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * @author sy
 * @date 2024/7/1 19:22
 */
public class Detection {
    private Boxes boxes = null;
    private float thresh = 0.5f;

    DetModel detModel = null;

    long rawImgHeight;
    long rawImgWidth;
    float ratioH;
    float ratioW;

    public Boxes getBoxes() {
        return boxes;
    }

    public void setBoxes(Boxes boxes) {
        this.boxes = boxes;
    }

    public float getThresh() {
        return thresh;
    }

    public void setThresh(float thresh) {
        this.thresh = thresh;
    }

    public Detection(DetModel detModel) {
        this.detModel = detModel;
    }

    public Detection(DetModel detModel, float thresh) {
        this.detModel = detModel;
        this.thresh = thresh;
    }

    /**
     *
     * @param imgMat
     * @return
     * @throws OrtException
     */
    public Boxes detect(Mat imgMat) throws OrtException {
        Map<String, OnnxTensor> inputMap = this.prepareInput(imgMat);
        float[][] predictions = inference(inputMap);
        Boxes detBoxes = this.processOutput(predictions);
        return detBoxes;
    }

    /**
     * @param inputMap
     * @return
     * @throws OrtException
     */
    private float[][] inference(Map<String, OnnxTensor> inputMap) throws OrtException {
        OrtSession.Result result = detModel.getSession().run(inputMap);
        float[][] predicitons = ((float[][][][])result.get(0).getValue())[0][0];
        return predicitons;
    }

    /**
     * @param img
     * @return
     * @throws OrtException
     */
    private Map<String, OnnxTensor> prepareInput(Mat img) throws OrtException {
        this.rawImgHeight = img.height();
        this.rawImgWidth = img.width();
        Mat inputImg = new Mat();

        float ratio = 1.0f;
        if (Math.max(rawImgHeight, rawImgWidth) > 960) {
            if (rawImgHeight > rawImgWidth) {
                ratio = (float) 960 / (float) rawImgHeight;
            } else {
                ratio = (float) 960 / (float) rawImgWidth;
            }
        }
        int resize_h = (int) (rawImgHeight * ratio);
        int resize_w = (int) (rawImgWidth * ratio);

        resize_h = Math.round((float) resize_h / 32f) * 32;
        resize_w = Math.round((float) resize_w / 32f) * 32;

        this.ratioH = resize_h / (float) rawImgHeight;
        this.ratioW = resize_w / (float) rawImgWidth;

        Imgproc.resize(img, inputImg, new Size(resize_w, resize_h));

        inputImg.convertTo(inputImg, CvType.CV_32FC3, 1./255);

        Imgproc.cvtColor(inputImg, inputImg, Imgproc.COLOR_BGR2RGB);

        Map<String, OnnxTensor> inputMap = CollectionUtil.newHashMap();
        float[] whc = new float[resize_h * resize_w * 3];

        inputImg.get(0, 0, whc);
        float[] chw = ImageUtils.whc2cwh(whc);

        FloatBuffer inputBuffer= FloatBuffer.wrap(chw);
        OnnxTensor inputTensor = OnnxTensor.createTensor(detModel.getEnv(), inputBuffer, new long[]{1, 3, resize_h, resize_w});
        inputMap.put(detModel.getInputName(), inputTensor);
        inputImg.release();
        return inputMap;
    }

    /**
     * @param predictions
     * @return
     */
    private Boxes processOutput(float[][] predictions) {
        NDManager ndManager = NDManager.newBaseManager();
        NDArray ndArray = ndManager.create(predictions);
        NDArray segmentation = ndArray.gt(this.thresh).toType(DataType.INT8, true);  // thresh=0.3 .mul(255f)
        Mat newMask = new Mat();
        Mat srcMat = NDArrayUtils.uint8NDArrayToMat(segmentation);
        Scalar scalar = new Scalar(255);
        Core.multiply(srcMat, scalar, newMask);
        srcMat.release();

        Pair boxScorePair = DetUtils.boxesFromBitmap(ndManager, ndArray, newMask, 0.5f);

        NDArray boxes = (NDArray) boxScorePair.getLeft();
        float[] scoreList = (float[]) boxScorePair.getRight();

        Boxes textBoxes = null;
        if(Optional.ofNullable(boxes).isPresent()) {
            NDList detBoxes = null;
            //boxes[:, :, 0] = boxes[:, :, 0] / ratio_w
            NDArray ndArray1 = boxes.get(":, :, 0").div(this.ratioW);
            boxes.set(new NDIndex(":, :, 0"), ndArray1);

            //boxes[:, :, 1] = boxes[:, :, 1] / ratio_h
            NDArray ndArray2 = boxes.get(":, :, 1").div(this.ratioH);
            boxes.set(new NDIndex(":, :, 1"), ndArray2);

            detBoxes = DetUtils.filterTagDetRes(boxes, this.rawImgWidth, this.rawImgHeight);

            detBoxes.detach();

            List<Box> boxList = CollectionUtil.newArrayList();

            for (var index=0; index< detBoxes.size(); index++) {
                NDArray box = detBoxes.get(index);
                float[] scoreBox = new float[4];
                scoreBox[0] = box.getFloat(0, 0);
                scoreBox[1] = box.getFloat(0, 1);
                scoreBox[2] = box.getFloat(2, 0);
                scoreBox[3] = box.getFloat(2, 1);
                float score = scoreList[index];
                Box TBox = new Box(scoreBox, score);
                boxList.add(TBox);
            }

            textBoxes = new Boxes(boxList);
        }

        // release Mat
        newMask.release();
//        mask.release();
//        structImage.release();

        return textBoxes;
    }

    /**
     * @param bbox
     */
    public void rescaleBoxes(float[] bbox) {
        bbox[0] /= detModel.getInputWidth();
        bbox[0] *= this.rawImgWidth;
        bbox[1] /= detModel.getInputHeight();
        bbox[1] *= this.rawImgHeight;
        bbox[2] /= detModel.getInputWidth();
        bbox[2] *= this.rawImgWidth;
        bbox[3] /= detModel.getInputHeight();
        bbox[3] *= this.rawImgHeight;
    }
}


