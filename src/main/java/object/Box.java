package object;

/**
 * @author sy
 * @date 2024/7/2 19:55
 */
public class Box {
    private float[] linePosition;
    private float score;

    public float[] getLinePosition() {
        return linePosition;
    }

    public float getScore() {
        return this.score;
    }

    public void setLinePosition(float[] linePosition) {
        this.linePosition = linePosition;
    }

    public Box(float[] linePosition, float score) {
        this.linePosition = linePosition;
        this.score = score;
    }

    @Override
    public String toString() {
        return "lineposition: " + linePosition[0] + "," + linePosition[1] + "," + linePosition[2] + "," + linePosition[3] + ", score: " + this.getScore();
    }
}
