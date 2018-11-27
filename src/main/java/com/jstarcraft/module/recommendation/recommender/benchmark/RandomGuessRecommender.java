package com.jstarcraft.module.recommendation.recommender.benchmark;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.recommendation.recommender.AbstractRecommender;

/**
 * 随机分数推荐器
 * 
 * @author Birdy
 *
 */
@ModelDefinition(value = { "userDimension", "itemDimension", "numberOfItems", "minimumOfScore", "maximumOfScore" })
public class RandomGuessRecommender extends AbstractRecommender {

	@Override
	protected void doPractice() {
	}

	@Override
	public synchronized float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		RandomUtility.setSeed(userIndex * numberOfItems + itemIndex);
		return RandomUtility.randomFloat(minimumOfScore, maximumOfScore);
	}

}
