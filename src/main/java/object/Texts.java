package object;


import java.util.List;

/**
 * @author sy
 * @date 2024/7/1 19:23
 */

public class Texts {
    private List<Text> texts;

    public List<Text> getTexts() {
        return texts;
    }

    public void setTexts(List<Text> texts) {
        this.texts = texts;
    }

    public Texts(List<Text> texts) {
        this.texts = texts;
    }

    @Override
    public String toString() {
        return "texts:" + this.getTexts();
    }
}
