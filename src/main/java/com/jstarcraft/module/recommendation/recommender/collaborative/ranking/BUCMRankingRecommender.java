package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.Map.Entry;

import com.jstarcraft.module.recommendation.recommender.collaborative.BUCMRecommender;

/**
 * Bayesian UCM: Nicola Barbieri et al., <strong>Modeling Item Selection and
 * Relevance for Accurate Recommendations: a Bayesian Approach</strong>, RecSys
 * 2011.
 * <p>
 * Thank the paper authors for providing source code and for having valuable
 * discussion.
 *
 * @author Guo Guibing and Haidong Zhang
 */
public class BUCMRankingRecommender extends BUCMRecommender {

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float value = 0F;
		for (int topicIndex = 0; topicIndex < numberOfFactors; ++topicIndex) {
			float sum = 0F;
			for (Entry<Float, Integer> term : scoreIndexes.entrySet()) {
				double rate = term.getKey();
				if (rate > meanOfScore) {
					sum += topicItemRateProbabilities[topicIndex][itemIndex][term.getValue()];
				}
			}
			value += userTopicProbabilities.getValue(userIndex, topicIndex) * topicItemProbabilities.getValue(topicIndex, itemIndex) * sum;
		}
		return value;
	}

}
