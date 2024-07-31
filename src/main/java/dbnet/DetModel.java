package dbnet;

import ai.onnxruntime.OrtException;
import object.Model;

/**
 * @author sy
 * @date 2024/7/1 20:20
 */
public class DetModel extends Model {
    public DetModel(String modelPath) throws OrtException {
        super(modelPath);
    }
}
