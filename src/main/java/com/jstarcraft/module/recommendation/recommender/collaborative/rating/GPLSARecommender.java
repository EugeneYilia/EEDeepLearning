package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.Gaussian;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixCollector;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.message.VarianceMessage;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorCollector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.ProbabilisticGraphicalRecommender;

/**
 * Thomas Hofmann, <strong>Collaborative Filtering via Gaussian Probabilistic
 * Latent Semantic Analysis</strong>, SIGIR 2003. <br>
 * <p>
 * <strong>Tempered EM:</strong> Thomas Hofmann, <strong>Unsupervised Learning
 * by Probabilistic Latent Semantic Analysis</strong>, Machine Learning, 42,
 * 177�C196, 2001.
 */
public class GPLSARecommender extends ProbabilisticGraphicalRecommender {

	/*
	 * {user, item, {topic z, probability}}
	 */
	protected Table<Integer, Integer, float[]> probabilityTensor;
	/*
	 * Conditional Probability: P(z|u)
	 */
	protected DenseMatrix userTopicProbabilities;
	/*
	 * Conditional Probability: P(v|y,z)
	 */
	protected DenseMatrix itemMus, itemSigmas;
	/*
	 * regularize ratings
	 */
	protected DenseVector userMus, userSigmas;
	/*
	 * smoothing weight
	 */
	protected float smoothWeight;
	/*
	 * tempered EM parameter beta, suggested by Wu Bin
	 */
	protected float beta;
	/*
	 * small value for initialization
	 */
	protected static float smallValue = MathUtility.EPSILON;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		// Initialize users' conditional probabilities
		userTopicProbabilities = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			DenseVector probabilityVector = userTopicProbabilities.getRowVector(userIndex);
			probabilityVector.normalize((index, value, message) -> {
				// 防止为0
				return RandomUtility.randomInteger(numberOfFactors) + 1;
			});
		}

		float mean = trainMatrix.getSum(false) / trainMatrix.getElementSize();
		VarianceMessage variance = new VarianceMessage(mean);
		trainMatrix.collectValues(MatrixCollector.ACCUMULATOR, variance, MathCalculator.PARALLEL);
		double standardDeviation = Math.sqrt(variance.getValue() / trainMatrix.getElementSize());

		userMus = DenseVector.valueOf(numberOfUsers);
		userSigmas = DenseVector.valueOf(numberOfUsers);
		smoothWeight = configuration.getInteger("rec.recommender.smoothWeight");
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			SparseVector userVector = trainMatrix.getRowVector(userIndex);
			int size = userVector.getElementSize();
			if (size < 1) {
				continue;
			}
			float mu = (userVector.getSum(false) + smoothWeight * mean) / (size + smoothWeight);
			userMus.setValue(userIndex, mu);
			variance = new VarianceMessage(mu);
			userVector.collectValues(VectorCollector.ACCUMULATOR, variance, MathCalculator.SERIAL);
			float sigma = variance.getValue();
			sigma += smoothWeight * Math.pow(standardDeviation, 2);
			sigma = (float) Math.sqrt(sigma / (size + smoothWeight));
			userSigmas.setValue(userIndex, sigma);
		}

		// Initialize Q
		// TODO 重构
		probabilityTensor = HashBasedTable.create();

		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			float rate = term.getValue();
			rate = (rate - userMus.getValue(userIndex)) / userSigmas.getValue(userIndex);
			term.setValue(rate);
			probabilityTensor.put(userIndex, itemIndex, new float[numberOfFactors]);
		}

		itemMus = DenseMatrix.valueOf(numberOfItems, numberOfFactors);
		itemSigmas = DenseMatrix.valueOf(numberOfItems, numberOfFactors);
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			SparseVector itemVector = trainMatrix.getColumnVector(itemIndex);
			int size = itemVector.getElementSize();
			if (size < 1) {
				continue;
			}
			float mu = itemVector.getSum(false) / itemVector.getElementSize();
			variance = new VarianceMessage(mu);
			itemVector.collectValues(VectorCollector.ACCUMULATOR, variance, MathCalculator.SERIAL);
			float sigma = variance.getValue();
			sigma = (float) Math.sqrt(sigma / size);
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				itemMus.setValue(itemIndex, topicIndex, mu + smallValue * RandomUtility.randomFloat(1F));
				itemSigmas.setValue(itemIndex, topicIndex, sigma + smallValue * RandomUtility.randomFloat(1F));
			}
		}
	}

	@Override
	protected void eStep() {
		// variational inference to compute Q
		float[] numerators = new float[numberOfFactors];
		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			float rate = term.getValue();
			float denominator = 0F;
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				float pdf = Gaussian.probabilityDensity(rate, itemMus.getValue(itemIndex, topicIndex), itemSigmas.getValue(itemIndex, topicIndex));
				float value = (float) Math.pow(userTopicProbabilities.getValue(userIndex, topicIndex) * pdf, beta); // Tempered
				// EM
				numerators[topicIndex] = value;
				denominator += value;
			}
			float[] probabilities = probabilityTensor.get(userIndex, itemIndex);
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				float probability = (denominator > 0 ? numerators[topicIndex] / denominator : 0);
				probabilities[topicIndex] = probability;
			}
		}
	}

	@Override
	protected void mStep() {
		float[] numerators = new float[numberOfFactors];
		// theta_u,z
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			SparseVector userVector = trainMatrix.getRowVector(userIndex);
			if (userVector.getElementSize() < 1) {
				continue;
			}
			float denominator = 0F;
			for (VectorScalar term : userVector) {
				int itemIndex = term.getIndex();
				float[] probabilities = probabilityTensor.get(userIndex, itemIndex);
				for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
					numerators[topicIndex] = probabilities[topicIndex];
					denominator += numerators[topicIndex];
				}
			}
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				userTopicProbabilities.setValue(userIndex, topicIndex, numerators[topicIndex] / denominator);
			}
		}

		// topicItemMu, topicItemSigma
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			SparseVector itemVector = trainMatrix.getColumnVector(itemIndex);
			if (itemVector.getElementSize() < 1) {
				continue;
			}
			float numerator = 0F, denominator = 0F;
			for (VectorScalar term : itemVector) {
				int userIndex = term.getIndex();
				float rate = term.getValue();
				float[] probabilities = probabilityTensor.get(userIndex, itemIndex);
				for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
					float probability = probabilities[topicIndex];
					numerator += rate * probability;
					denominator += probability;
				}
			}
			float mu = denominator > 0F ? numerator / denominator : 0F;
			numerator = 0F;
			for (VectorScalar term : itemVector) {
				int userIndex = term.getIndex();
				float rate = term.getValue();
				float[] probabilities = probabilityTensor.get(userIndex, itemIndex);
				for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
					double probability = probabilities[topicIndex];
					numerator += Math.pow(rate - mu, 2) * probability;
				}
			}
			float sigma = (float) (denominator > 0F ? Math.sqrt(numerator / denominator) : 0F);
			for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
				itemMus.setValue(itemIndex, topicIndex, mu);
				itemSigmas.setValue(itemIndex, topicIndex, sigma);
			}
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float sum = 0F;
		for (int topicIndex = 0; topicIndex < numberOfFactors; topicIndex++) {
			sum += userTopicProbabilities.getValue(userIndex, topicIndex) * itemMus.getValue(itemIndex, topicIndex);
		}
		return userMus.getValue(userIndex) + userSigmas.getValue(userIndex) * sum;
	}

}
