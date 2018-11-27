package com.jstarcraft.module.recommendation.recommender.content;

/**
 * EFM Recommender Zhang Y, Lai G, Zhang M, et al. Explicit factor models for
 * explainable recommendation based on phrase-level sentiment analysis[C]
 * {@code Proceedings of the 37th international ACM SIGIR conference on Research & development in information retrieval.  ACM, 2014: 83-92}.
 *
 * @author ChenXu and SunYatong
 */
public class EFMRatingRecommender extends EFMRecommender {

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float value = predict(userIndex, itemIndex);
		if (value < minimumOfScore)
			return minimumOfScore;
		if (value > maximumOfScore)
			return maximumOfScore;
		return value;
	}

}
