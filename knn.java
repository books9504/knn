import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

public class knn {
	public static int kValue = 3;
	public static double[] output;
	public static double[] distanceValues;
	public static int[] kSmallestIndex;
	static double[] prediction;

	public static void main(String[] args) throws FileNotFoundException, Exception {
		Matrix trainData = new Matrix();
		trainData.loadArff("datasets\\1trainingSet.arff");

		Matrix testData = new Matrix();
		testData.loadArff("datasets\\1testSet.arff");

		// normalize the input features

		trainData.normalize();
		testData.normalize();

		Matrix features = new Matrix(trainData, 0, 0, trainData.rows(), trainData.cols() - 1);
		Matrix labels = new Matrix(trainData, 0, trainData.cols() - 1, trainData.rows(), 1);
		Matrix testFeatures = new Matrix(testData, 0, 0, testData.rows(), testData.cols() - 1);
		Matrix testLabels = new Matrix(testData, 0, testData.cols() - 1, testData.rows(), 1);

		distanceValues = new double[features.rows()];
		output = new double[testLabels.rows()];
		for (int i = 0; i < testFeatures.rows(); i++) {
			distanceValues = calcDistance(features, testFeatures.row(i));
			HashMap<Integer, Double> distanceMap = new HashMap<Integer, Double>();
			for (int j = 0; j < distanceValues.length; j++) {
				distanceMap.put(j, distanceValues[j]);
			}
			Comparator<Integer> comparator = new ValueComparator<Integer, Double>(distanceMap);
			TreeMap<Integer, Double> sorted = new TreeMap<Integer, Double>(comparator);
			sorted.putAll(distanceMap);
			// end smallest dist
			// get output by majority votes
			//output[i] = getMajorityVotes(sorted, labels);

			// get output by weighed votes
			output[i]= getWeighedVotes(sorted,labels);

			// get output for regression by no distance weighing
			// output[i]= getMeanValues(sorted,labels);

			// get output for regression by distance weighing
			// output[i]= getWeighedMeanValues(sorted,labels);
			// System.out.println("the pred is: "+ output[i]);
		}
		double acc = 0;
		acc = calculateAcc(output, testLabels);
		System.out.println("Accuracy is: " + acc);
		System.out.println("Rmse is: " + calculateRmse(output, testLabels));

	}

	private static double getMeanValues(TreeMap<Integer, Double> sorted, Matrix labels) {
		// TODO Auto-generated method stub
		prediction = new double[kValue];
		double sum = 0;
		double mean = 0;
		Set<Integer> keys = sorted.keySet();
		int count = 0;
		for (int key : keys) {
			// 5,4,2
			prediction[count] = labels.get(key, 0);
			count++;
			if (count > (kValue - 1))
				break;
		}
		for (int i = 0; i < kValue; i++) {
			sum += prediction[i];
		}
		mean = sum / kValue;
		return mean;
	}

	private static double getWeighedMeanValues(TreeMap<Integer, Double> sorted, Matrix labels) {
		Set<Integer> keys = sorted.keySet();
		int count = 0;
		double sum = 0;
		double wSum = 0;
		prediction = new double[kValue];
		double[] weighedDistance = new double[kValue];
		for (int key : keys) {
			// 5,4,2
			prediction[count] = labels.get(key, 0);
			weighedDistance[count] = (double) 1 / (sorted.get(key) * sorted.get(key));
			count++;
			if (count > (kValue - 1))
				break;
		}
		for (int i = 0; i < kValue; i++) {
			sum += prediction[i] * weighedDistance[i];
			wSum += weighedDistance[i];
		}
		double net = sum / wSum;
		return net;
	}

	private static double calculateRmse(double[] pred, Matrix labels) {
		double mse = 0.0;
		double[] delta = new double[pred.length];
		for (int i = 0; i < pred.length; i++) {
			delta[i] = pred[i] - labels.get(i, 0);
			mse += delta[i] * delta[i];
		}
		double finalMse = mse / (double) pred.length;
		return Math.sqrt(finalMse);
	}

	private static double calculateAcc(double[] pred, Matrix labels) {
		// TODO Auto-generated method stub
		int correct = 0;
		for (int i = 0; i < pred.length; i++) {
			if (pred[i] == labels.get(i, 0)) {
				correct++;
			}
		}
		return (double) correct / pred.length;
	}

	private static double getMajorityVotes(TreeMap<Integer, Double> sorted, Matrix labels) {
		// TODO Auto-generated method stub
		prediction = new double[kValue];
		Set<Integer> keys = sorted.keySet();
		int count = 0, o1 = 0, o2 = 0;
		for (int key : keys) {
			// 5,4,2
			prediction[count] = labels.get(key, 0);
			count++;
			if (count > (kValue - 1))
				break;
		}
		for (int i = 0; i < kValue; i++) {
			if (prediction[i] == 0.0)
				o1++;
			else if (prediction[i] == 1.0)
				o2++;
		}

		return (o1 >= o2) ? 0.0 : 1.0;
	}

	private static double getWeighedVotes(TreeMap<Integer, Double> sorted, Matrix labels) {
		Set<Integer> keys = sorted.keySet();
		int count = 0;
		double[] weighedDistance = new double[2];

		for (int key : keys) {
			if (labels.get(key, 0) == 0.0) {
				weighedDistance[0] += (double) 1 / (sorted.get(key) * sorted.get(key));
			} else if (labels.get(key, 0) == 1.0) {
				weighedDistance[1] += (double) 1 / (sorted.get(key) * sorted.get(key));
			}
			count++;
			if (count > (kValue - 1))
				break;
		}
		return (weighedDistance[0] >= weighedDistance[1]) ? 0.0 : 0.1;
	}

	private static double[] calcDistance(Matrix features, double[] test) {
		// TODO Auto-generated method stub
		double[] manhattanDistance = new double[features.rows()];
		// double[] kLeastDistance= new double[kValue];
		for (int i = 0; i < features.rows(); i++) {
			for (int j = 0; j < features.cols(); j++) {

				manhattanDistance[i] += Math.abs(features.get(i, j) - test[j]);// distance
																				// of
			}
		}

		return manhattanDistance;
	}

}
