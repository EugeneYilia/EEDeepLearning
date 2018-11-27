package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.Arrays;
import java.util.Comparator;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.algorithm.Probability;
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
public class LambdaFMStaticRecommender extends LambdaFMRecommender {

	// Static
	private float staticRho;
	protected Probability itemProbabilities;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		staticRho = configuration.getFloat("rec.item.distribution.parameter");
		// calculate popularity
		Integer[] orderItems = new Integer[numberOfItems];
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			orderItems[itemIndex] = itemIndex;
		}
		Arrays.sort(orderItems, new Comparator<Integer>() {
			@Override
			public int compare(Integer leftItemIndex, Integer rightItemIndex) {
				return (trainMatrix.getColumnScope(leftItemIndex) > trainMatrix.getColumnScope(rightItemIndex) ? -1 : (trainMatrix.getColumnScope(leftItemIndex) < trainMatrix.getColumnScope(rightItemIndex) ? 1 : 0));
			}
		});
		Integer[] itemOrders = new Integer[numberOfItems];
		for (int index = 0; index < numberOfItems; index++) {
			int itemIndex = orderItems[index];
			itemOrders[itemIndex] = index;
		}
		itemProbabilities = new Probability(numberOfItems, (index, value, message) -> {
			return (float) Math.exp(-(itemOrders[index] + 1) / (numberOfItems * staticRho));
		});
	}

	@Override
	protected float getGradientValue(DefaultScalar scalar, int[] dataPaginations, int[] dataPositions) {
		int userIndex;
		while (true) {
			userIndex = RandomUtility.randomInteger(numberOfUsers);
			SparseVector userVector = trainMatrix.getRowVector(userIndex);
			if (userVector.getElementSize() == 0 || userVector.getElementSize() == numberOfItems) {
				continue;
			}

			int from = dataPaginations[userIndex], to = dataPaginations[userIndex + 1];
			int positivePosition = dataPositions[RandomUtility.randomInteger(from, to)];
			for (int index = 0; index < negativeKeys.length; index++) {
				positiveKeys[index] = trainTensor.getIndex(index, positivePosition);
			}

			// TODO 注意,此处为了故意制造负面特征.
			Integer negativeItemIndex = null;
			while (negativeItemIndex == null) {
				negativeItemIndex = itemProbabilities.random(userVector);
			}
			int negativePosition = dataPositions[RandomUtility.randomInteger(from, to)];
			for (int index = 0; index < negativeKeys.length; index++) {
				negativeKeys[index] = trainTensor.getIndex(index, negativePosition);
			}
			negativeKeys[itemDimension] = negativeItemIndex;
			break;
		}

		positiveVector = getFeatureVector(positiveKeys);
		negativeVector = getFeatureVector(negativeKeys);

		float positiveScore = predict(scalar, positiveVector);
		float negativeScore = predict(scalar, negativeVector);

		float error = positiveScore - negativeScore;

		// 由于pij_real默认为1,所以简化了loss的计算.
		// loss += -pij_real * Math.log(pij) - (1 - pij_real) *
		// Math.log(1 - pij);
		totalLoss += (float) -Math.log(MathUtility.logistic(error));
		float gradient = calaculateGradientValue(lossType, error);
		return gradient;
	}

}
