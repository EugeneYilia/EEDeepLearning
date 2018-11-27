package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * Biased Matrix Factorization Recommender
 *
 * @author GuoGuibing and Keqiang Wang
 */
public class BiasedMFRecommender extends MatrixFactorizationRecommender {
	/**
	 * bias regularization
	 */
	protected float regBias;

	/**
	 * user biases
	 */
	protected DenseVector userBiases;

	/**
	 * user biases
	 */
	protected DenseVector itemBiases;

	/*
	 * (non-Javadoc)
	 *
	 * @see net.librec.recommender.AbstractRecommender#setup()
	 */
	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		regBias = configuration.getFloat("rec.bias.regularization", 0.01F);

		// initialize the userBiased and itemBiased
		userBiases = DenseVector.valueOf(numberOfUsers, VectorMapper.distributionOf(distribution));
		itemBiases = DenseVector.valueOf(numberOfItems, VectorMapper.distributionOf(distribution));
	}

	@Override
	protected void doPractice() {
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;

			for (MatrixScalar term : trainMatrix) {
				int userIndex = term.getRow(); // user userIdx
				int itemIndex = term.getColumn(); // item itemIdx
				float rate = term.getValue(); // real rating on item
												// itemIdx rated by user
												// userIdx
				float predict = predict(userIndex, itemIndex);
				float error = rate - predict;
				totalLoss += error * error;

				// update user and item bias
				float userBias = userBiases.getValue(userIndex);
				userBiases.shiftValue(userIndex, learnRate * (error - regBias * userBias));
				totalLoss += regBias * userBias * userBias;
				float itemBias = itemBiases.getValue(itemIndex);
				itemBiases.shiftValue(itemIndex, learnRate * (error - regBias * itemBias));
				totalLoss += regBias * itemBias * itemBias;

				// update user and item factors
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactor = userFactors.getValue(userIndex, factorIndex);
					float itemFactor = itemFactors.getValue(itemIndex, factorIndex);
					userFactors.shiftValue(userIndex, factorIndex, learnRate * (error * itemFactor - userRegularization * userFactor));
					itemFactors.shiftValue(itemIndex, factorIndex, learnRate * (error * userFactor - itemRegularization * itemFactor));
					totalLoss += userRegularization * userFactor * userFactor + itemRegularization * itemFactor * itemFactor;
				}
			}

			totalLoss *= 0.5D;
			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			isLearned(iterationStep);
			currentLoss = totalLoss;
		}
	}

	@Override
	protected float predict(int userIndex, int itemIndex) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		DenseVector userVector = userFactors.getRowVector(userIndex);
		DenseVector itemVector = itemFactors.getRowVector(itemIndex);
		float value = scalar.dotProduct(userVector, itemVector).getValue();
		value += meanOfScore + userBiases.getValue(userIndex) + itemBiases.getValue(itemIndex);
		return value;
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		return predict(userIndex, itemIndex);
	}

}
