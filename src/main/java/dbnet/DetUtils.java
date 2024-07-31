package dbnet;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import utils.common.CollectionUtil;
import utils.cv.NDArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sy
 * @date 2024/7/1 20:20
 */
public class DetUtils {

    /**
     * sort the points based on their x-coordinates
     * @param pts
     * @return
     */
    private static NDArray orderPointsClockwise(NDArray pts) {
        var list = new NDList();
        long[] indexes = pts.get(":, 0").argSort().toLongArray();

        // grab the left-most and right-most points from the sorted
        // x-roodinate points
        var s1 = pts.getShape();
        var leftMost1 = pts.get(indexes[0] + ",:");
        var leftMost2 = pts.get(indexes[1] + ",:");
        var leftMost = leftMost1.concat(leftMost2).reshape(2, 2);
        var rightMost1 = pts.get(indexes[2] + ",:");
        var rightMost2 = pts.get(indexes[3] + ",:");
        var rightMost = rightMost1.concat(rightMost2).reshape(2, 2);

        // now, sort the left-most coordinates according to their
        // y-coordinates so we can grab the top-left and bottom-left
        // points, respectively
        indexes = leftMost.get(":, 1").argSort().toLongArray();
        var lt = leftMost.get(indexes[0] + ",:");
        var lb = leftMost.get(indexes[1] + ",:");
        indexes = rightMost.get(":, 1").argSort().toLongArray();
        var rt = rightMost.get(indexes[0] + ",:");
        var rb = rightMost.get(indexes[1] + ",:");

        list.add(lt);
        list.add(rt);
        list.add(rb);
        list.add(lb);

        var rect = NDArrays.concat(list).reshape(4, 2);
        return rect;
    }

    /**
     * get boxes from the binarized image predicted by DB
     * @param manager
     * @param pred    the binarized image predicted by DB.
     * @param mat    new 'pred' after threshold filtering.
     * @param boxThresh
     */
    public static Pair boxesFromBitmap(NDManager manager, NDArray pred, Mat mat, float boxThresh) {
        var destHeight = (int) pred.getShape().get(0);
        var destWidth = (int) pred.getShape().get(1);
        var height = mat.rows();
        var width = mat.cols();

        List<MatOfPoint> contours = CollectionUtil.newArrayList();
        var hierarchy = new Mat();

        // find contours
        Imgproc.findContours(mat,
                             contours,
                             hierarchy,
                             Imgproc.RETR_LIST,
                             Imgproc.CHAIN_APPROX_SIMPLE,
                             new Point(0, 0));

        System.out.println("contours: " + contours.size());

        var numContours = Math.min(contours.size(), 1000);

        var boxNDList = new NDList();
        float[] scoreList = new float[numContours];
        for (int index = 0; index < numContours; index++) {
            MatOfPoint contour = contours.get(index);
            float[][] pointsArr = new float[4][2];
            var sside = getMiniBoxes(contour, pointsArr);
            if (sside < 3) {
                continue;
            }

            var points = manager.create(pointsArr);

            var score = boxScoreFast(manager, pred, points);
            if (score < boxThresh) {
                continue;
            }

            var box = unClip(manager, points); // TODO get_mini_boxes(box)

            MatOfPoint matOfPoint = new MatOfPoint(
                    new Point(box.getFloat(0, 0), box.getFloat(0, 1)),
                    new Point(box.getFloat(1, 0), box.getFloat(1, 1)),
                    new Point(box.getFloat(2, 0), box.getFloat(2, 1)),
                    new Point(box.getFloat(3, 0), box.getFloat(3, 1))
            );

            sside = getMiniBoxes(matOfPoint, pointsArr);
            matOfPoint.release();
            if(sside < 3 + 2) {
                continue;
            }

            // box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            var boxes1 = box.get(":,0").div(width).mul(destWidth).round().clip(0, destWidth);
            box.set(new NDIndex(":, 0"), boxes1);
            // box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            var boxes2 = box.get(":,1").div(height).mul(destHeight).round().clip(0, destHeight);
            box.set(new NDIndex(":, 1"), boxes2);

            boxNDList.add(box);
//          boxes.set(new NDIndex(count + ",:,:"), box);
            scoreList[index] = score;

            // release memory
            contour.release();

        }
//        if (count < num_contours) {
//            NDArray newBoxes = manager.zeros(new Shape(count, 4, 2), DataType.FLOAT32);
//            newBoxes.set(new NDIndex("0,0,0"), boxes.get(":" + count + ",:,:"));
//            boxes = newBoxes;
//        }


        Pair pair = null;
        if((boxNDList.size() > 0) && (scoreList.length > 0)) {
            NDArray boxes = NDArrays.stack(boxNDList);
            pair = Pair.of(boxes, scoreList);
        }


        // release
        hierarchy.release();

        return pair;
    }

    /**
     * shrink or expand the boxaccording to 'unclip_ratio'
     * @param points The predicted box.
     * @return uncliped box
     */
    public static NDArray unClip(NDManager manager, NDArray points) {
        points = orderPointsClockwise(points);
        float[] pointsArr = points.toFloatArray();
        float[] lt = java.util.Arrays.copyOfRange(pointsArr, 0, 2);
        float[] lb = java.util.Arrays.copyOfRange(pointsArr, 6, 8);

        float[] rt = java.util.Arrays.copyOfRange(pointsArr, 2, 4);
        float[] rb = java.util.Arrays.copyOfRange(pointsArr, 4, 6);

        var width = distance(lt, rt);
        var height = distance(lt, lb);

        if (width > height) {
            var k = (lt[1] - rt[1]) / (lt[0] - rt[0]); // y = k * x + b

            var deltaDis = height;
            var deltaX = (float) Math.sqrt((deltaDis * deltaDis) / (k * k + 1));
            var deltaY = Math.abs(k * deltaX);

            if (k > 0) {
                pointsArr[0] = lt[0] - deltaX + deltaY;
                pointsArr[1] = lt[1] - deltaY - deltaX;
                pointsArr[2] = rt[0] + deltaX + deltaY;
                pointsArr[3] = rt[1] + deltaY - deltaX;

                pointsArr[4] = rb[0] + deltaX - deltaY;
                pointsArr[5] = rb[1] + deltaY + deltaX;
                pointsArr[6] = lb[0] - deltaX - deltaY;
                pointsArr[7] = lb[1] - deltaY + deltaX;
            } else {
                pointsArr[0] = lt[0] - deltaX - deltaY;
                pointsArr[1] = lt[1] + deltaY - deltaX;
                pointsArr[2] = rt[0] + deltaX - deltaY;
                pointsArr[3] = rt[1] - deltaY - deltaX;

                pointsArr[4] = rb[0] + deltaX + deltaY;
                pointsArr[5] = rb[1] - deltaY + deltaX;
                pointsArr[6] = lb[0] - deltaX + deltaY;
                pointsArr[7] = lb[1] + deltaY + deltaX;
            }
        } else {
            var k = (lt[1] - rt[1]) / (lt[0] - rt[0]); // y = k * x + b

            var deltaDis = width;
            var deltaY = (float) Math.sqrt((deltaDis * deltaDis) / (k * k + 1));
            var deltaX = Math.abs(k * deltaY);

            if (k > 0) {
                pointsArr[0] = lt[0] + deltaX - deltaY;
                pointsArr[1] = lt[1] - deltaY - deltaX;
                pointsArr[2] = rt[0] + deltaX + deltaY;
                pointsArr[3] = rt[1] - deltaY + deltaX;

                pointsArr[4] = rb[0] - deltaX + deltaY;
                pointsArr[5] = rb[1] + deltaY + deltaX;
                pointsArr[6] = lb[0] - deltaX - deltaY;
                pointsArr[7] = lb[1] + deltaY - deltaX;
            } else {
                pointsArr[0] = lt[0] - deltaX - deltaY;
                pointsArr[1] = lt[1] - deltaY + deltaX;
                pointsArr[2] = rt[0] - deltaX + deltaY;
                pointsArr[3] = rt[1] - deltaY - deltaX;

                pointsArr[4] = rb[0] + deltaX + deltaY;
                pointsArr[5] = rb[1] + deltaY - deltaX;
                pointsArr[6] = lb[0] + deltaX - deltaY;
                pointsArr[7] = lb[1] + deltaY + deltaX;
            }
        }
        points = manager.create(pointsArr).reshape(4, 2);

        return points;
    }

    /**
     * distance between point1 and point2
     * @param point1
     * @param point2
     * @return
     */
    private static float distance(float[] point1, float[] point2) {
        var disX = point1[0] - point2[0];
        var disY = point1[1] - point2[1];
        var dis = (float) Math.sqrt(disX * disX + disY * disY);
        return dis;
    }

    /**
     * get boxes from the contour or box.
     * @param contour   The predicted contour.
     * @param pointsArr The predicted box.
     * @return smaller side of box
     */
    private static int getMiniBoxes(MatOfPoint contour, float[][] pointsArr) {
        // https://blog.csdn.net/qq_37385726/article/details/82313558
        // bounding_box[1] - rect returns the length and width of the rectangle
//        MatOfPoint2f contour2f = ImageUtils.matOfPointToMatOfPoint2f(contour);
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        var rect = Imgproc.minAreaRect(contour2f);

        var points = new Mat();
        Imgproc.boxPoints(rect, points);

        float[][] fourPoints = new float[4][2];
        for(int row=0; row<fourPoints.length; row++) {
            fourPoints[row][0] = (float) points.get(row, 0)[0];
            fourPoints[row][1] = (float) points.get(row, 1)[0];
        }

        float[] tmpPoint = new float[2];
        for (int i = 0; i < fourPoints.length; i++) {
            for (int j = i + 1; j < fourPoints.length; j++) {
                if (fourPoints[j][0] < fourPoints[i][0]) {
                    tmpPoint[0] = fourPoints[i][0];
                    tmpPoint[1] = fourPoints[i][1];
                    fourPoints[i][0] = fourPoints[j][0];
                    fourPoints[i][1] = fourPoints[j][1];
                    fourPoints[j][0] = tmpPoint[0];
                    fourPoints[j][1] = tmpPoint[1];
                }
            }
        }

        var index1 = 0;
        var index2 = 1;
        var index3 = 2;
        var index4 = 3;

        if (fourPoints[1][1] > fourPoints[0][1]) {
            index1 = 0;
            index4 = 1;
        } else {
            index1 = 1;
            index4 = 0;
        }

        if (fourPoints[3][1] > fourPoints[2][1]) {
            index2 = 2;
            index3 = 3;
        } else {
            index2 = 3;
            index3 = 2;
        }

        pointsArr[0] = fourPoints[index1];
        pointsArr[1] = fourPoints[index2];
        pointsArr[2] = fourPoints[index3];
        pointsArr[3] = fourPoints[index4];

        var height = rect.boundingRect().height;
        var width = rect.boundingRect().width;
        var sside = Math.min(height, width);

        // release
        points.release();
        contour2f.release();
        return sside;
    }

    /**
     * calculate the score of box.
     * @param bitmap The binarized image predicted by DB.
     * @param points The predicted box
     * @return
     */
    public static float boxScoreFast(NDManager manager, NDArray bitmap, NDArray points) {
        var box = points.get(":");
        var h = bitmap.getShape().get(0);
        var w = bitmap.getShape().get(1);
        // xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        var xmin = box.get(":, 0").min().floor().clip(0, w - 1).toType(DataType.INT32, true).toIntArray()[0];
        var xmax = box.get(":, 0").max().ceil().clip(0, w - 1).toType(DataType.INT32, true).toIntArray()[0];
        var ymin = box.get(":, 1").min().floor().clip(0, h - 1).toType(DataType.INT32, true).toIntArray()[0];
        var ymax = box.get(":, 1").max().ceil().clip(0, h - 1).toType(DataType.INT32, true).toIntArray()[0];

        var mask = manager.zeros(new Shape(ymax - ymin + 1, xmax - xmin + 1), DataType.UINT8);

        box.set(new NDIndex(":, 0"), box.get(":, 0").sub(xmin));
        box.set(new NDIndex(":, 1"), box.get(":, 1").sub(ymin));

        //mask - convert from NDArray to Mat
        Mat maskMat = NDArrayUtils.uint8NDArrayToMat(mask);

        //mask - convert from NDArray to Mat - 4 rows, 2 cols
        Mat boxMat = NDArrayUtils.floatNDArrayToMat(box, CvType.CV_32S);

//        boxMat.reshape(1, new int[]{1, 4, 2});
        List<MatOfPoint> pts = new ArrayList<>();
        MatOfPoint matOfPoint = NDArrayUtils.matToMatOfPoint(boxMat); // new MatOfPoint(boxMat);
        pts.add(matOfPoint);
        Imgproc.fillPoly(maskMat, pts, new Scalar(1));

        NDArray subBitMap = bitmap.get(ymin + ":" + (ymax + 1) + "," + xmin + ":" + (xmax + 1));
        Mat bitMapMat = NDArrayUtils.floatNDArrayToMat(subBitMap);

        Scalar score = Core.mean(bitMapMat, maskMat);
        float scoreValue = (float) score.val[0];
        // release
        maskMat.release();
        boxMat.release();
        bitMapMat.release();
        return scoreValue;
    }

    public static NDList filterTagDetRes(NDArray dt_boxes, long width, long height) {
        NDList boxesList = new NDList();

        int num = (int) dt_boxes.getShape().get(0);
        for (int i = 0; i < num; i++) {
            NDArray box = dt_boxes.get(i);
            box = orderPointsClockwise(box);
            box = clipDetRes(box, width, height);
            float[] box0 = box.get(0).toFloatArray();
            float[] box1 = box.get(1).toFloatArray();
            float[] box3 = box.get(3).toFloatArray();
            int rect_width = (int) Math.sqrt(Math.pow(box1[0] - box0[0], 2) + Math.pow(box1[1] - box0[1], 2));
            int rect_height = (int) Math.sqrt(Math.pow(box3[0] - box0[0], 2) + Math.pow(box3[1] - box0[1], 2));
            if (rect_width <= 3 || rect_height <= 3) {
                continue;
            }
            boxesList.add(box);
        }
        return boxesList;
    }

    public static NDArray clipDetRes(NDArray points, long width, long height) {
        for (int i = 0; i < points.getShape().get(0); i++) {
            int value = Math.max((int) points.get(i, 0).toFloatArray()[0], 0);
            value = (int) Math.min(value, width - 1);
            points.set(new NDIndex(i + ",0"), value);
            value = Math.max((int) points.get(i, 1).toFloatArray()[0], 0);
            value = (int) Math.min(value, height - 1);
            points.set(new NDIndex(i + ",1"), value);
        }

        return points;
    }

    /**
     * @param matrix
     * @return
     */
    public static float[][] transposeMatrix(float[][] matrix) {
        float[][] transMatrix = new float[matrix[0].length][matrix.length];
        for(int i=0; i<matrix.length; i++) {
            for(int j=0; j<matrix[0].length; j++) {
                transMatrix[j][i] = matrix[i][j];
            }
        }
        return transMatrix;
    }

    /**
     * @param probabilities
     * @return
     */
    public static int predMax(float[] probabilities) {
        float maxVal = Float.NEGATIVE_INFINITY;
        int idx = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxVal) {
                maxVal = probabilities[i];
                idx = i;
            }
        }
        return idx;
    }

}
