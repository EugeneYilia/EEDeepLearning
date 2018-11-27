package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.ProbabilisticGraphicalRecommender;

/**
 * Thomas Hofmann, <strong>Latent semantic models for collaborative
 * filtering</strong>, ACM Transactions on Information Systems. 2004. <br>
 *
 * @author Haidong Zhang and Keqiang Wang
 */

public class PLSARecommender extends ProbabilisticGraphicalRecommender {

	/**
	 * {user, item, {topic z, probability}}
	 */
	private Table<Integer, Integer, DenseVector> probabilityTensor;

	/**
	 * Conditional Probability: P(z|u)
	 */
	private DenseMatrix userTopicProbabilities, userTopicSums;

	/**
	 * Conditional Probability: P(i|z)
	 */
	private DenseMatrix topicItemProbabilities, topicItemSums;

	/**
	 * topic probability sum value
	 */
	private DenseVector topicProbabilities;

	/**
	 * entry[u]: number of tokens rated by user u.
	 */
	private DenseVector userRateTimes;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);

		// TODO 此处代码可以消除(使用常量Marker代替或者使用binarize.threshold)
		for (MatrixScalar term : trainMatrix) {
			term.setValue(1F);
		}

		userTopicSums = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
		topicItemSums = DenseMatrix.valueOf(numberOfFactors, numberOfItems);
		topicProbabilities = DenseVector.valueOf(numberOfFactors);

		userTopicProbabilities = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			DenseVector probabilityVector = userTopicProbabilities.getRowVector(userIndex);
			probabilityVector.normalize((index, value, message) -> {
				// 防止为0
				return RandomUtility.randomInteger(numberOfFactors) + 1;
			});
		}

		topicItemProbabilities = DenseMatrix.valueOf(numberOfFactors, numberOfItems);
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			DenseVector probabilityVector = topicItemProbabilities.getRowVector(topicIndex);
			probabilityVector.normalize((index, value, message) -> {
				// 防止为0
				return RandomUtility.randomInteger(numberOfItems) + 1;
			});
		}

		// initialize Q

		// initialize Q
		probabilityTensor = HashBasedTable.create();
		userRateTimes = DenseVector.valueOf(numberOfUsers);
		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			probabilityTensor.put(userIndex, itemIndex, DenseVector.valueOf(numberOfFactors));
			userRateTimes.shiftValue(userIndex, term.getValue());
		}
	}

	@Override
	protected void eStep() {
		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			DenseVector probabilities = probabilityTensor.get(userIndex, itemIndex);
			probabilities.normalize((index, value, message) -> {
				return userTopicProbabilities.getValue(userIndex, index) * topicItemProbabilities.getValue(index, itemIndex);
			});
		}
	}

	@Override
	protected void mStep() {
		userTopicSums.mapValues(MatrixMapper.ZERO, null, MathCalculator.PARALLEL);
		topicItemSums.mapValues(MatrixMapper.ZERO, null, MathCalculator.PARALLEL);
		topicProbabilities.mapValues(VectorMapper.ZERO, null, MathCalculator.SERIAL);
		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			float numerator = term.getValue();
			DenseVector probabilities = probabilityTensor.get(userIndex, itemIndex);
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				float value = probabilities.getValue(topicIndex) * numerator;
				userTopicSums.shiftValue(userIndex, topicIndex, value);
				topicItemSums.shiftValue(topicIndex, itemIndex, value);
				topicProbabilities.shiftValue(topicIndex, value);
			}
		}

		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			float denominator = userRateTimes.getValue(userIndex);
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				float value = denominator > 0F ? userTopicSums.getValue(userIndex, topicIndex) / denominator : 0F;
				userTopicProbabilities.setValue(userIndex, topicIndex, value);
			}
		}

		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			float probability = topicProbabilities.getValue(topicIndex);
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				float value = probability > 0F ? topicItemSums.getValue(topicIndex, itemIndex) / probability : 0F;
				topicItemProbabilities.setValue(topicIndex, itemIndex, value);
			}
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		DenseVector userVector = userTopicProbabilities.getRowVector(userIndex);
		DenseVector itemVector = topicItemProbabilities.getColumnVector(itemIndex);
		return scalar.dotProduct(userVector, itemVector).getValue();
	}

}
