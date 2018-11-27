package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
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
 * @author Haidong Zhang and Keqiang Wang
 */

public class AspectModelRankingRecommender extends ProbabilisticGraphicalRecommender {

	/**
	 * Conditional distribution: P(u|z)
	 */
	private DenseMatrix userProbabilities, userSums;

	/**
	 * Conditional distribution: P(i|z)
	 */
	private DenseMatrix itemProbabilities, itemSums;

	/**
	 * topic distribution: P(z)
	 */
	private DenseVector topicProbabilities, topicSums;

	private DenseVector probabilities;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);

		// Initialize topic distribution
		// TODO 考虑重构
		topicProbabilities = DenseVector.valueOf(numberOfFactors);
		topicSums = DenseVector.valueOf(numberOfFactors);
		topicProbabilities.normalize((index, value, message) -> {
			// 防止为0
			return RandomUtility.randomInteger(numberOfFactors) + 1;
		});

		userProbabilities = DenseMatrix.valueOf(numberOfFactors, numberOfUsers);
		userSums = DenseMatrix.valueOf(numberOfFactors, numberOfUsers);
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			DenseVector probabilityVector = userProbabilities.getRowVector(topicIndex);
			probabilityVector.normalize((index, value, message) -> {
				// 防止为0
				return RandomUtility.randomInteger(numberOfUsers) + 1;
			});
		}

		itemProbabilities = DenseMatrix.valueOf(numberOfFactors, numberOfItems);
		itemSums = DenseMatrix.valueOf(numberOfFactors, numberOfItems);
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			DenseVector probabilityVector = itemProbabilities.getRowVector(topicIndex);
			probabilityVector.normalize((index, value, message) -> {
				// 防止为0
				return RandomUtility.randomInteger(numberOfItems) + 1;
			});
		}

		probabilities = DenseVector.valueOf(numberOfFactors);
	}

	/*
	 *
	 */
	@Override
	protected void eStep() {
		topicSums.mapValues(VectorMapper.ZERO, null, MathCalculator.SERIAL);
		userSums.mapValues(MatrixMapper.ZERO, null, MathCalculator.PARALLEL);
		itemSums.mapValues(MatrixMapper.ZERO, null, MathCalculator.PARALLEL);
		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			probabilities.normalize((index, value, message) -> {
				return userProbabilities.getValue(index, userIndex) * itemProbabilities.getValue(index, itemIndex) * topicProbabilities.getValue(index);
			});
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				float value = probabilities.getValue(topicIndex) * term.getValue();
				topicSums.shiftValue(topicIndex, value);
				userSums.shiftValue(topicIndex, userIndex, value);
				itemSums.shiftValue(topicIndex, itemIndex, value);
			}
		}
	}

	@Override
	protected void mStep() {
		float scale = 1F / topicSums.getSum(false);
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			topicProbabilities.setValue(topicIndex, topicSums.getValue(topicIndex) * scale);
			float userSum = userProbabilities.getColumnVector(topicIndex).getSum(false);
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				userProbabilities.setValue(topicIndex, userIndex, userSums.getValue(topicIndex, userIndex) / userSum);
			}
			float itemSum = itemProbabilities.getColumnVector(topicIndex).getSum(false);
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				itemProbabilities.setValue(topicIndex, itemIndex, itemSums.getValue(topicIndex, itemIndex) / itemSum);
			}
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float value = 0F;
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			value += userProbabilities.getValue(topicIndex, userIndex) * itemProbabilities.getValue(topicIndex, itemIndex) * topicProbabilities.getValue(topicIndex);
		}
		return value;
	}

}
