package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.tensor.TensorScalar;
import com.jstarcraft.module.math.structure.vector.ArrayVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.FactorizationMachineRecommender;

/**
 * Field-aware Factorization Machines Yuchin Juan, "Field Aware Factorization
 * Machines for CTR Prediction", 10th ACM Conference on Recommender Systems,
 * 2016
 *
 * @author Li Wenxi and Tan Jiale
 */

public class FFMRecommender extends FactorizationMachineRecommender {
	/**
	 * learning rate of stochastic gradient descent
	 */
	private float learnRate;
	/**
	 * record the <feature: filed>
	 */
	private int[] featureDimensions;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);

		// Matrix for p * (factor * filed)
		// TODO 此处应该还是稀疏
		featureFactors = DenseMatrix.valueOf(numberOfFeatures, numberOfFactors * trainTensor.getOrderSize(), MatrixMapper.distributionOf(distribution));

		int size = 0;
		for (int dimension = 0; dimension < trainTensor.getOrderSize(); dimension++) {
			size += trainTensor.getDimensionSize(dimension);
		}

		// TODO FeatureDimension应该取消
		featureDimensions = new int[size];
		// init the map for feature of filed

		int count = 0;
		for (int dimension = 0; dimension < trainTensor.getOrderSize(); dimension++) {
			size = trainTensor.getDimensionSize(dimension);
			for (int index = 0; index < size; index++) {
				featureDimensions[count + index] = dimension;
			}
			count += size;
		}

		learnRate = configuration.getFloat("rec.iterator.learnRate");
	}

	@Override
	protected void doPractice() {
		DefaultScalar scalar = DefaultScalar.getInstance();
		for (int iterationStep = 0; iterationStep < numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			int outerIndex = 0;
			int innerIndex = 0;
			float outerValue = 0F;
			float innerValue = 0F;
			float oldWeight = 0F;
			float newWeight = 0F;
			float oldFactor = 0F;
			float newFactor = 0F;
			int[] keys = new int[trainTensor.getOrderSize()];
			for (TensorScalar tensorTerm : trainTensor) {
				tensorTerm.getIndexes(keys);
				// TODO 因为每次的data都是1,可以考虑避免重复构建featureVector.
				ArrayVector featureVector = getFeatureVector(keys);
				float rate = tensorTerm.getValue();
				float predict = predict(scalar, featureVector);
				float error = predict - rate;
				totalLoss += error * error;

				// global bias
				totalLoss += biasRegularization * globalBias * globalBias;

				// update w0
				float hW0 = 1;
				float gradW0 = error * hW0 + biasRegularization * globalBias;
				globalBias += -learnRate * gradW0;

				// 1-way interactions
				for (VectorScalar outerTerm : featureVector) {
					outerIndex = outerTerm.getIndex();
					innerIndex = 0;
					oldWeight = weightVector.getValue(outerIndex);
					newWeight = outerTerm.getValue();
					newWeight = error * newWeight + weightRegularization * oldWeight;
					weightVector.shiftValue(outerIndex, -learnRate * newWeight);
					totalLoss += weightRegularization * oldWeight * oldWeight;
					outerValue = outerTerm.getValue();
					innerValue = 0F;
					// 2-way interactions
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						oldFactor = featureFactors.getValue(outerIndex, featureDimensions[outerIndex] + factorIndex);
						newFactor = 0F;
						for (VectorScalar innerTerm : featureVector) {
							innerIndex = innerTerm.getIndex();
							innerValue = innerTerm.getValue();
							if (innerIndex != outerIndex) {
								newFactor += outerValue * featureFactors.getValue(innerIndex, featureDimensions[outerIndex] + factorIndex) * innerValue;
							}
						}
						newFactor = error * newFactor + factorRegularization * oldFactor;
						featureFactors.shiftValue(outerIndex, featureDimensions[outerIndex] + factorIndex, -learnRate * newFactor);
						totalLoss += factorRegularization * oldFactor * oldFactor;
					}
				}
			}

			totalLoss *= 0.5;
			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			currentLoss = totalLoss;
		}
	}

	@Override
	protected float predict(DefaultScalar scalar, ArrayVector featureVector) {
		float value = 0F;
		// global bias
		value += globalBias;
		// 1-way interaction
		value += scalar.dotProduct(weightVector, featureVector).getValue();
		int outerIndex = 0;
		int innerIndex = 0;
		float outerValue = 0F;
		float innerValue = 0F;
		// 2-way interaction
		for (int featureIndex = 0; featureIndex < numberOfFactors; featureIndex++) {
			for (VectorScalar outerVector : featureVector) {
				outerIndex = outerVector.getIndex();
				outerValue = outerVector.getValue();
				for (VectorScalar innerVector : featureVector) {
					innerIndex = innerVector.getIndex();
					innerValue = innerVector.getValue();
					if (outerIndex != innerIndex) {
						value += featureFactors.getValue(outerIndex, featureDimensions[innerIndex] + featureIndex) * featureFactors.getValue(innerIndex, featureDimensions[outerIndex] + featureIndex) * outerValue * innerValue;
					}
				}
			}
		}

		if (value > maximumOfScore) {
			value = maximumOfScore;
		}
		if (value < minimumOfScore) {
			value = minimumOfScore;
		}
		return value;
	}

}
