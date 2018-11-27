package com.jstarcraft.module.recommendation.recommender.content;

import java.util.Arrays;
import java.util.Comparator;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.recommendation.configure.Configuration;

/**
 * EFM Recommender Zhang Y, Lai G, Zhang M, et al. Explicit factor models for
 * explainable recommendation based on phrase-level sentiment analysis[C]
 * {@code Proceedings of the 37th international ACM SIGIR conference on Research & development in information retrieval.  ACM, 2014: 83-92}.
 *
 * @author ChenXu and SunYatong
 */
public class EFMRankingRecommender extends EFMRecommender {

	private float threshold;

	private int featureLimit;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		threshold = configuration.getFloat("efmranking.threshold", 1F);
		featureLimit = configuration.getInteger("efmranking.featureLimit", 250);
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];

		// TODO 此处可以优化性能
		Integer[] orderIndexes = new Integer[numberOfFeatures];
		for (int featureIndex = 0; featureIndex < numberOfFeatures; featureIndex++) {
			orderIndexes[featureIndex] = featureIndex;
		}
		MathVector vector = DenseVector.valueOf(numberOfFeatures);
		vector.dotProduct(userExplicitFactors.getRowVector(userIndex), featureFactors, true, MathCalculator.SERIAL);
		Arrays.sort(orderIndexes, new Comparator<Integer>() {
			@Override
			public int compare(Integer leftIndex, Integer rightIndex) {
				return (vector.getValue(leftIndex) > vector.getValue(rightIndex) ? -1 : (vector.getValue(leftIndex) < vector.getValue(rightIndex) ? 1 : 0));
			}
		});

		float value = 0F;
		for (int index = 0; index < featureLimit; index++) {
			int featureIndex = orderIndexes[index];
			value += predictUserFactor(scalar, userIndex, featureIndex) * predictItemFactor(scalar, itemIndex, featureIndex);
		}
		value = threshold * (value / (featureLimit * maximumOfScore));
		value = value + (1F - threshold) * predict(userIndex, itemIndex);
		return value;
	}

}
