package object;

import java.util.List;

/**
 * @author sy
 * @date 2024/7/1 19:33
 */
public class Boxes {
    private List<Box> boxes;

    public List<Box> getBoxes() {
        return boxes;
    }

    public void setBoxes(List<Box> boxes) {
        this.boxes = boxes;
    }

    public Boxes(List<Box> boxes) {
        this.boxes = boxes;
    }

    @Override
    public String toString() {
        return "boxes:" + this.getBoxes();
    }
}
