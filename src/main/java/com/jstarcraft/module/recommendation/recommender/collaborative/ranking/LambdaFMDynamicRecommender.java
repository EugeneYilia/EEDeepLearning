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
import com.jstarcraft.module.math.structure.vector.ArrayVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.recommendation.configure.Configuration;

/**
 * 
 * YUAN et al., <strong>LambdaFM: Learning Optimal Ranking with Factorization
 * Machines Using Lambda Surrogates</strong>, CIKM 2016.
 * 
 * @author fajie yuan
 */
public class LambdaFMDynamicRecommender extends LambdaFMRecommender {

	// Dynamic
	private float dynamicRho;
	private int numberOfOrders;
	private Probability orderProbabilities;
	private int[][] negativeIndexes;
	private float[] negativeValues;
	private Integer[] orderIndexes;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		dynamicRho = configuration.getFloat("rec.item.distribution.parameter");
		numberOfOrders = configuration.getInteger("rec.number.orders", 10);

		orderProbabilities = new Probability(numberOfOrders, (index, value, message) -> {
			return (float) (Math.exp(-(index + 1) / (numberOfOrders * dynamicRho)));
		});
		negativeIndexes = new int[numberOfOrders][trainTensor.getOrderSize()];
		negativeValues = new float[numberOfOrders];
		orderIndexes = new Integer[numberOfOrders];
		for (int index = 0; index < numberOfOrders; index++) {
			orderIndexes[index] = index;
		}
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
			// TODO negativeGroup.size()可能永远达不到numberOfNegatives,需要处理
			for (int orderIndex = 0; orderIndex < numberOfOrders; orderIndex++) {
				int negativeItemIndex = RandomUtility.randomInteger(numberOfItems - userVector.getElementSize());
				for (int position = 0, size = userVector.getElementSize(); position < size; position++) {
					if (negativeItemIndex >= userVector.getIndex(position)) {
						negativeItemIndex++;
						continue;
					}
					break;
				}
				negativeKeys = negativeIndexes[orderIndex];
				// TODO 注意,此处为了故意制造负面特征.
				int negativePosition = dataPositions[RandomUtility.randomInteger(from, to)];
				for (int index = 0; index < negativeKeys.length; index++) {
					negativeKeys[index] = trainTensor.getIndex(index, negativePosition);
				}
				negativeKeys[itemDimension] = negativeItemIndex;
				ArrayVector vector = getFeatureVector(negativeKeys);
				negativeValues[orderIndex] = predict(scalar, vector);
			}

			int orderIndex = orderProbabilities.random();
			Arrays.sort(orderIndexes, new Comparator<Integer>() {
				@Override
				public int compare(Integer leftIndex, Integer rightIndex) {
					return (negativeValues[leftIndex] > negativeValues[rightIndex] ? -1 : (negativeValues[leftIndex] < negativeValues[rightIndex] ? 1 : 0));
				}
			});
			negativeKeys = negativeIndexes[orderIndexes[orderIndex]];
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
