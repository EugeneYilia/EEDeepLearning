package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * <ul>
 * <li><strong>PMF:</strong> Ruslan Salakhutdinov and Andriy Mnih, Probabilistic
 * Matrix Factorization, NIPS 2008.</li>
 * <li><strong>RegSVD:</strong> Arkadiusz Paterek, <strong>Improving Regularized
 * Singular Value Decomposition</strong> Collaborative Filtering, Proceedings of
 * KDD Cup and Workshop, 2007.</li>
 * </ul>
 *
 * @author guoguibin and zhanghaidong
 */
public class PMFRecommender extends MatrixFactorizationRecommender {

	@Override
	protected void doPractice() {
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			for (MatrixScalar term : trainMatrix) {
				int userIndex = term.getRow(); // user
				int itemIndex = term.getColumn(); // item
				float rate = term.getValue();
				float predict = predict(userIndex, itemIndex);
				float error = rate - predict;
				totalLoss += error * error;

				// update factors
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactor = userFactors.getValue(userIndex, factorIndex), itemFactor = itemFactors.getValue(itemIndex, factorIndex);
					userFactors.shiftValue(userIndex, factorIndex, learnRate * (error * itemFactor - userRegularization * userFactor));
					itemFactors.shiftValue(itemIndex, factorIndex, learnRate * (error * userFactor - itemRegularization * itemFactor));
					totalLoss += userRegularization * userFactor * userFactor + itemRegularization * itemFactor * itemFactor;
				}
			}

			totalLoss *= 0.5F;
			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			isLearned(iterationStep);
			currentLoss = totalLoss;
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float value = super.predict(userIndex, itemIndex);
		if (value > maximumOfScore) {
			value = maximumOfScore;
		} else if (value < minimumOfScore) {
			value = minimumOfScore;
		}
		return value;
	}

}
