package object;

import ai.onnxruntime.*;

import java.util.Map;
import java.util.Optional;

/**
 * @author sy
 * @date 2024/7/1 20:25
 */
public class Model {
    private OrtSession session;
    private OrtEnvironment env;

    private int gpuDeviceId = -1;

    private long inputHeight;
    private long inputWidth;

    private int numInputElements;
    private String inputName;
    private String outputName;
    private long[] inputShape;
    private OnnxJavaType inputType;


    public OrtSession getSession() {
        return session;
    }

    public OrtEnvironment getEnv() {
        return env;
    }

    public int getGpuDeviceId() {
        return gpuDeviceId;
    }

    public long getInputHeight() {
        return inputHeight;
    }

    public long getInputWidth() {
        return inputWidth;
    }

    public int getNumInputElements() {
        return numInputElements;
    }

    public String getInputName() {
        return inputName;
    }

    public String getOutputName() {
        return outputName;
    }

    public long[] getInputShape() {
        return inputShape;
    }

    public OnnxJavaType getInputType() {
        return inputType;
    }

    public Model(String path) throws OrtException {
        initModel(path);
    }

    public Model(String path, int gpuDeviceId) throws OrtException {
        this.gpuDeviceId = gpuDeviceId;
        initModel(path);
    }

    /**
     * @param path
     * @throws OrtException
     */
    private void initModel(String path) throws OrtException {
        System.out.println("init and load model...");
        this.env = OrtEnvironment.getEnvironment();
        var sessionOptions = new OrtSession.SessionOptions();
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        if(this.gpuDeviceId >= 0) {
            sessionOptions.addCPU(false);
            sessionOptions.addCUDA(gpuDeviceId);
        } else {
            sessionOptions.addCPU(true);
        }
        this.session = this.env.createSession(path, sessionOptions);

        Map<String, NodeInfo> inputMetaMap = this.session.getInputInfo();
        this.inputName = this.session.getInputNames().iterator().next();
        NodeInfo inputMeta = inputMetaMap.get(this.inputName);
        this.inputType = ((TensorInfo)inputMeta.getInfo()).type;
        this.inputShape = ((TensorInfo) inputMeta.getInfo()).getShape();

        this.numInputElements = (int) (this.inputShape[1] * this.inputShape[2] * this.inputShape[3]);
        this.inputHeight = this.inputShape[2];
        this.inputWidth = this.inputShape[3];
        this.outputName = this.session.getOutputNames().iterator().next();
    }

    public void closeModel() {
        System.out.println("close model...");
        if (Optional.ofNullable(this.session).isPresent()) {
            try {
                this.session.close();
            } catch (OrtException e) {
                e.printStackTrace();
            }
        }
        if(Optional.ofNullable(this.env).isPresent()) {
            this.env.close();
        }
    }
}
