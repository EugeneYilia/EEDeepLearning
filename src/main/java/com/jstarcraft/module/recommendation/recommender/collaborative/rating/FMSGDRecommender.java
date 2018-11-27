package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.tensor.TensorScalar;
import com.jstarcraft.module.math.structure.vector.ArrayVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.FactorizationMachineRecommender;

/**
 * Stochastic Gradient Descent with Square Loss Rendle, Steffen, "Factorization
 * Machines", Proceedings of the 10th IEEE International Conference on Data
 * Mining, 2010 Rendle, Steffen, "Factorization Machines with libFM", ACM
 * Transactions on Intelligent Systems and Technology, 2012
 *
 * @author Jiaxi Tang and Ma Chen
 */
public class FMSGDRecommender extends FactorizationMachineRecommender {
	/**
	 * learning rate of stochastic gradient descent
	 */
	private float learnRate;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		learnRate = configuration.getFloat("rec.iterator.learnRate");
	}

	@Override
	protected void doPractice() {
		DefaultScalar scalar = DefaultScalar.getInstance();
		for (int iterationStep = 0; iterationStep < numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
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

				// TODO 因为此处相当与迭代trainTensor的featureVector,所以hW0才会是1D.
				float hW0 = 1F;
				float bias = error * hW0 + biasRegularization * globalBias;

				// update w0
				globalBias += -learnRate * bias;

				// 1-way interactions
				for (VectorScalar outerTerm : featureVector) {
					int outerIndex = outerTerm.getIndex();
					float oldWeight = weightVector.getValue(outerIndex);
					float featureWeight = outerTerm.getValue();
					float newWeight = error * featureWeight + weightRegularization * oldWeight;
					weightVector.shiftValue(outerIndex, -learnRate * newWeight);
					totalLoss += weightRegularization * oldWeight * oldWeight;
					// 2-way interactions
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						float oldValue = featureFactors.getValue(outerIndex, factorIndex);
						float newValue = 0F;
						for (VectorScalar innerTerm : featureVector) {
							int innerIndex = innerTerm.getIndex();
							if (innerIndex != outerIndex) {
								newValue += featureWeight * featureFactors.getValue(innerIndex, factorIndex) * innerTerm.getValue();
							}
						}
						newValue = error * newValue + factorRegularization * oldValue;
						featureFactors.shiftValue(outerIndex, factorIndex, -learnRate * newValue);
						totalLoss += factorRegularization * oldValue * oldValue;
					}
				}
			}

			totalLoss *= 0.5F;
			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			currentLoss = totalLoss;
		}
	}

	@Override
	protected float predict(DefaultScalar scalar, ArrayVector featureVector) {
		float value = super.predict(scalar, featureVector);

		if (value > maximumOfScore) {
			value = maximumOfScore;
		}
		if (value < minimumOfScore) {
			value = minimumOfScore;
		}
		return value;
	}

}