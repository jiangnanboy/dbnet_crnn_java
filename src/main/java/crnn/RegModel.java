package crnn;

import ai.onnxruntime.OrtException;
import object.Model;
import utils.common.CollectionUtil;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

/**
 * @author sy
 * @date 2024/7/5 21:25
 */
public class RegModel extends Model {
    private Map<Long, String> labelMapping = CollectionUtil.newHashMap();
    public RegModel (String modelPath, String vocabPath) throws OrtException {
        super(modelPath);
        loadVocab(vocabPath);
    }

    /**
     * @param vocabPath
     */
    public void loadVocab(String vocabPath) {
        System.out.println("load vocab...");
        try(BufferedReader br = Files.newBufferedReader(Paths.get(vocabPath), Charset.forName("utf-8"))) {
            String line = null;
            long cnt = 1L;
            while(null != (line = br.readLine())) {
                line = line.trim();
                labelMapping.put(cnt, line);
                cnt ++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * @return
     */
    public Map<Long, String> getLabelMapping() {
        return this.labelMapping;
    }
}
