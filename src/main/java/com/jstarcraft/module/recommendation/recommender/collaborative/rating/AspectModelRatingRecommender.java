package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.Gaussian;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.ProbabilisticGraphicalRecommender;

/**
 * <h3>Latent class models for collaborative filtering</h3>
 * <p>
 * This implementation refers to the method proposed by Thomas et al. at IJCAI
 * 1999.
 * <p>
 * <strong>Tempered EM:</strong> Thomas Hofmann, <strong>Latent class models for
 * collaborative filtering </strong>, IJCAI. 1999, 99: 688-693.
 *
 * @author guoguibin and Haidong Zhang
 */
public class AspectModelRatingRecommender extends ProbabilisticGraphicalRecommender {
	/*
	 * Conditional distribution: P(u|z)
	 */
	private DenseMatrix userProbabilities, userSums;
	/*
	 * Conditional distribution: P(i|z)
	 */
	private DenseMatrix itemProbabilities, itemSums;
	/*
	 * topic distribution: P(z)
	 */
	private DenseVector topicProbabilities, topicSums;
	/*
	 *
	 */
	private DenseVector meanProbabilities, meanSums;
	/*
	 *
	 */
	private DenseVector varianceProbabilities, varianceSums;

	/*
	 * small value
	 */
	private static float smallValue = MathUtility.EPSILON;
	/*
	 * {user, item, {topic z, probability}}
	 */
	private float[][] probabilityTensor;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		// Initialize topic distribution
		topicProbabilities = DenseVector.valueOf(numberOfFactors);
		topicProbabilities.normalize((index, value, message) -> {
			// 防止为0
			return RandomUtility.randomInteger(numberOfFactors) + 1;
		});
		topicSums = DenseVector.valueOf(numberOfFactors);

		// intialize conditional distribution P(u|z)
		userProbabilities = DenseMatrix.valueOf(numberOfFactors, numberOfUsers);
		userSums = DenseMatrix.valueOf(numberOfFactors, numberOfUsers);
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			DenseVector probabilityVector = userProbabilities.getRowVector(topicIndex);
			probabilityVector.normalize((index, value, message) -> {
				// 防止为0
				return RandomUtility.randomInteger(numberOfUsers) + 1;
			});
		}

		// initialize conditional distribution P(i|z)
		itemProbabilities = DenseMatrix.valueOf(numberOfFactors, numberOfItems);
		itemSums = DenseMatrix.valueOf(numberOfFactors, numberOfItems);
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			DenseVector probabilityVector = itemProbabilities.getRowVector(topicIndex);
			probabilityVector.normalize((index, value, message) -> {
				// 防止为0
				return RandomUtility.randomInteger(numberOfItems) + 1;
			});
		}

		// initialize Q
		probabilityTensor = new float[numberOfActions][numberOfFactors];

		float globalMean = trainMatrix.getSum(false) / trainMatrix.getElementSize();
		meanProbabilities = DenseVector.valueOf(numberOfFactors);
		varianceProbabilities = DenseVector.valueOf(numberOfFactors);
		meanSums = DenseVector.valueOf(numberOfFactors);
		varianceSums = DenseVector.valueOf(numberOfFactors);
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			meanProbabilities.setValue(topicIndex, globalMean);
			varianceProbabilities.setValue(topicIndex, 2);
		}
	}

	@Override
	protected void eStep() {
		topicSums.mapValues(VectorMapper.constantOf(smallValue), null, MathCalculator.SERIAL);
		userSums.mapValues(MatrixMapper.ZERO, null, MathCalculator.PARALLEL);
		itemSums.mapValues(MatrixMapper.ZERO, null, MathCalculator.PARALLEL);
		meanSums.mapValues(VectorMapper.ZERO, null, MathCalculator.SERIAL);
		varianceSums.mapValues(VectorMapper.constantOf(smallValue), null, MathCalculator.SERIAL);
		// variational inference to compute Q
		int actionIndex = 0;
		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			float denominator = 0F;
			float[] numerator = probabilityTensor[actionIndex++];
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				float value = topicProbabilities.getValue(topicIndex) * userProbabilities.getValue(topicIndex, userIndex) * itemProbabilities.getValue(topicIndex, itemIndex) * Gaussian.probabilityDensity(term.getValue(), meanProbabilities.getValue(topicIndex), varianceProbabilities.getValue(topicIndex));
				numerator[topicIndex] = value;
				denominator += value;
			}
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				float probability = denominator > 0 ? numerator[topicIndex] / denominator : 0F;
				numerator[topicIndex] = probability;
			}

			float score = term.getValue();
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				float probability = numerator[topicIndex];
				topicSums.shiftValue(topicIndex, probability);
				userSums.shiftValue(topicIndex, userIndex, probability);
				itemSums.shiftValue(topicIndex, itemIndex, probability);
				meanSums.shiftValue(topicIndex, score * probability);
			}
		}

		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			float mean = meanSums.getValue(topicIndex) / topicSums.getValue(topicIndex);
			meanProbabilities.setValue(topicIndex, mean);
		}

		actionIndex = 0;
		for (MatrixScalar term : trainMatrix) {
			float[] probabilities = probabilityTensor[actionIndex++];
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				float mean = meanProbabilities.getValue(topicIndex);
				float error = term.getValue() - mean;
				float probability = probabilities[topicIndex];
				varianceSums.shiftValue(topicIndex, error * error * probability);
			}
		}
	}

	@Override
	protected void mStep() {
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			varianceProbabilities.setValue(topicIndex, varianceSums.getValue(topicIndex) / topicSums.getValue(topicIndex));
			topicProbabilities.setValue(topicIndex, topicSums.getValue(topicIndex) / numberOfActions);
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				userProbabilities.setValue(topicIndex, userIndex, userSums.getValue(topicIndex, userIndex) / topicSums.getValue(topicIndex));
			}
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				itemProbabilities.setValue(topicIndex, itemIndex, itemSums.getValue(topicIndex, itemIndex) / topicSums.getValue(topicIndex));
			}
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float value = 0F;
		float denominator = 0F;
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			float weight = topicProbabilities.getValue(topicIndex) * userProbabilities.getValue(topicIndex, userIndex) * itemProbabilities.getValue(topicIndex, itemIndex);
			denominator += weight;
			value += weight * meanProbabilities.getValue(topicIndex);
		}
		value = value / denominator;
		if (Float.isNaN(value)) {
			value = meanOfScore;
		}
		return value;
	}

}
