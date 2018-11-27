package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;

/**
 * Yehuda Koren, <strong>Factorization Meets the Neighborhood: a Multifaceted
 * Collaborative Filtering Model.</strong>, KDD 2008. Asymmetric SVD++
 * Recommender
 *
 * @author Bin Wu(wubin@gs.zzu.edu.cn)
 */
public class ASVDPlusPlusRecommender extends BiasedMFRecommender {

	private DenseMatrix positiveFactors, negativeFactors;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		positiveFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.distributionOf(distribution));
		negativeFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.distributionOf(distribution));
	}

	@Override
	protected void doPractice() {
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			// TODO 目前没有totalLoss.
			totalLoss = 0f;
			for (MatrixScalar matrixTerm : trainMatrix) {
				int userIndex = matrixTerm.getRow();
				int itemIndex = matrixTerm.getColumn();
				float rate = matrixTerm.getValue();
				float predict = predict(userIndex, itemIndex);
				float error = rate - predict;
				SparseVector userVector = trainMatrix.getRowVector(userIndex);

				// update factors
				float userBiasValue = userBiases.getValue(userIndex);
				userBiases.shiftValue(userIndex, learnRate * (error - regBias * userBiasValue));
				float itemBiasValue = itemBiases.getValue(itemIndex);
				itemBiases.shiftValue(itemIndex, learnRate * (error - regBias * itemBiasValue));

				float squareRoot = (float) Math.sqrt(userVector.getElementSize());
				float[] positiveSums = new float[numberOfFactors];
				float[] negativeSums = new float[numberOfFactors];
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float positiveSum = 0F;
					float negativeSum = 0F;
					for (VectorScalar term : userVector) {
						int ItemIdx = term.getIndex();
						positiveSum += positiveFactors.getValue(ItemIdx, factorIndex);
						negativeSum += negativeFactors.getValue(ItemIdx, factorIndex) * (rate - meanOfScore - userBiases.getValue(userIndex) - itemBiases.getValue(ItemIdx));
					}
					positiveSums[factorIndex] = squareRoot > 0 ? positiveSum / squareRoot : positiveSum;
					negativeSums[factorIndex] = squareRoot > 0 ? negativeSum / squareRoot : negativeSum;
				}

				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactor = userFactors.getValue(userIndex, factorIndex);
					float itemFactor = itemFactors.getValue(itemIndex, factorIndex);
					float userValue = error * itemFactor - userRegularization * userFactor;
					float itemValue = error * (userFactor + positiveSums[factorIndex] + negativeSums[factorIndex]) - itemRegularization * itemFactor;
					userFactors.shiftValue(userIndex, factorIndex, learnRate * userValue);
					itemFactors.shiftValue(itemIndex, factorIndex, learnRate * itemValue);
					for (VectorScalar term : userVector) {
						int index = term.getIndex();
						float positiveFactor = positiveFactors.getValue(index, factorIndex);
						float negativeFactor = negativeFactors.getValue(index, factorIndex);
						float positiveDelta = error * itemFactor / squareRoot - userRegularization * positiveFactor;
						float negativeDelta = error * itemFactor * (rate - meanOfScore - userBiases.getValue(userIndex) - itemBiases.getValue(index)) / squareRoot - userRegularization * negativeFactor;
						positiveFactors.shiftValue(index, factorIndex, learnRate * positiveDelta);
						negativeFactors.shiftValue(index, factorIndex, learnRate * negativeDelta);
					}
				}
			}
		}
	}

	@Override
	protected float predict(int userIndex, int itemIndex) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		DenseVector userVector = userFactors.getRowVector(userIndex);
		DenseVector itemVector = itemFactors.getRowVector(itemIndex);
		float value = meanOfScore + userBiases.getValue(userIndex) + itemBiases.getValue(itemIndex) + scalar.dotProduct(userVector, itemVector).getValue();
		SparseVector rateVector = trainMatrix.getRowVector(userIndex);
		float squareRoot = (float) Math.sqrt(rateVector.getElementSize());
		for (VectorScalar term : rateVector) {
			itemIndex = term.getIndex();
			DenseVector positiveVector = positiveFactors.getRowVector(itemIndex);
			DenseVector negativeVector = negativeFactors.getRowVector(itemIndex);
			value += scalar.dotProduct(positiveVector, itemVector).getValue() / squareRoot;
			float scale = term.getValue() - meanOfScore - userBiases.getValue(userIndex) - itemBiases.getValue(itemIndex);
			value += scalar.dotProduct(negativeVector, itemVector).getValue() * scale / squareRoot;
		}
		if (Double.isNaN(value)) {
			value = meanOfScore;
		}
		return value;
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		return predict(userIndex, itemIndex);
	}

}