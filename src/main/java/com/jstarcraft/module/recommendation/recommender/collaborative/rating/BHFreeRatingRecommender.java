package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import java.util.Map.Entry;

import com.jstarcraft.module.recommendation.recommender.collaborative.BHFreeRecommender;

/**
 * Barbieri et al., <strong>Balancing Prediction and Recommendation Accuracy:
 * Hierarchical Latent Factors for Preference Data</strong>, SDM 2012. <br>
 * <p>
 * <strong>Remarks:</strong> this class implements the BH-free method.
 *
 * @author Guo Guibing and haidong zhang
 */
public class BHFreeRatingRecommender extends BHFreeRecommender {

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float value = 0F, probabilities = 0F;
		for (Entry<Float, Integer> entry : scoreIndexes.entrySet()) {
			float rate = entry.getKey();
			float probability = 0F;
			for (int userTopic = 0; userTopic < numberOfUserTopics; userTopic++) {
				for (int itemTopic = 0; itemTopic < numberOfItemTopics; itemTopic++) {
					probability += user2TopicProbabilities.getValue(userIndex, userTopic) * userTopic2ItemTopicProbabilities.getValue(userTopic, itemTopic) * userTopic2ItemTopicRateProbabilities[userTopic][itemTopic][entry.getValue()];
				}
			}
			value += rate * probability;
			probabilities += probability;
		}
		return value / probabilities;
	}

}
