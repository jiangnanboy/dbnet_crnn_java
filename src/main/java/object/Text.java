package object;

/**
 * @author sy
 * @date 2024/7/2 19:55
 */
public class Text {
    private String text;

    public String getText() {
        return text;
    }

    public Text(String text) {
        this.text = text;
    }

    @Override
    public String toString() {
        return "text: " + this.getText();
    }
}
