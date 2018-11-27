package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.recommendation.configure.Configuration;

/**
 * 
 * YUAN et al., <strong>LambdaFM: Learning Optimal Ranking with Factorization
 * Machines Using Lambda Surrogates</strong>, CIKM 2016.
 * 
 * @author fajie yuan
 */
public class LambdaFMWeightRecommender extends LambdaFMRecommender {

	// Weight
	private float[] orderLosses;
	private float epsilon;
	private int Y, N;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		epsilon = configuration.getFloat("epsilon");
		orderLosses = new float[numberOfItems - 1];
		float orderLoss = 0F;
		for (int orderIndex = 1; orderIndex < numberOfItems; orderIndex++) {
			orderLoss += 1F / orderIndex;
			orderLosses[orderIndex - 1] = orderLoss;
		}
		for (int rankIndex = 1; rankIndex < numberOfItems; rankIndex++) {
			orderLosses[rankIndex - 1] /= orderLoss;
		}
	}

	@Override
	protected float getGradientValue(DefaultScalar scalar, int[] dataPaginations, int[] dataPositions) {
		int userIndex;
		float positiveScore;
		float negativeScore;
		while (true) {
			userIndex = RandomUtility.randomInteger(numberOfUsers);
			SparseVector userVector = trainMatrix.getRowVector(userIndex);
			if (userVector.getElementSize() == 0 || userVector.getElementSize() == numberOfItems) {
				continue;
			}

			N = 0;
			Y = numberOfItems - trainMatrix.getRowScope(userIndex);
			int from = dataPaginations[userIndex], to = dataPaginations[userIndex + 1];
			int positivePosition = dataPositions[RandomUtility.randomInteger(from, to)];
			for (int index = 0; index < negativeKeys.length; index++) {
				positiveKeys[index] = trainTensor.getIndex(index, positivePosition);
			}
			positiveVector = getFeatureVector(positiveKeys);
			positiveScore = predict(scalar, positiveVector);
			do {
				N++;
				int negativeItemIndex = RandomUtility.randomInteger(numberOfItems - userVector.getElementSize());
				for (int position = 0, size = userVector.getElementSize(); position < size; position++) {
					if (negativeItemIndex >= userVector.getIndex(position)) {
						negativeItemIndex++;
						continue;
					}
					break;
				}
				// TODO 注意,此处为了故意制造负面特征.
				int negativePosition = dataPositions[RandomUtility.randomInteger(from, to)];
				// TODO 注意,此处为了故意制造负面特征.
				for (int index = 0; index < negativeKeys.length; index++) {
					negativeKeys[index] = trainTensor.getIndex(index, negativePosition);
				}
				negativeKeys[itemDimension] = negativeItemIndex;
				negativeVector = getFeatureVector(negativeKeys);
				negativeScore = predict(scalar, negativeVector);
			} while ((positiveScore - negativeScore > epsilon) && N < Y - 1);
			break;
		}

		float error = positiveScore - negativeScore;

		// 由于pij_real默认为1,所以简化了loss的计算.
		// loss += -pij_real * Math.log(pij) - (1 - pij_real) *
		// Math.log(1 - pij);
		totalLoss += (float) -Math.log(MathUtility.logistic(error));
		float gradient = calaculateGradientValue(lossType, error);
		int orderIndex = (int) ((Y - 1) / N);
		float orderLoss = orderLosses[orderIndex];
		gradient = gradient * orderLoss;
		return gradient;
	}

}
