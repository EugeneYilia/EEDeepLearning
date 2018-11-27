package com.jstarcraft.module.recommendation.recommender.benchmark;

import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.recommendation.recommender.AbstractRecommender;

/**
 * 全局平均分数推荐器
 * 
 * @author Birdy
 *
 */
@ModelDefinition(value = { "meanOfScore" })
public class GlobalAverageRecommender extends AbstractRecommender {

	@Override
	protected void doPractice() {
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		return meanOfScore;
	}

}
