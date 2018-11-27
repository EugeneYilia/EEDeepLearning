package com.jstarcraft.module.recommendation.recommender.benchmark;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.AbstractRecommender;

/**
 * 最受欢迎推荐器
 * 
 * @author Birdy
 *
 */
@ModelDefinition(value = { "itemDimension", "populars" })
public class MostPopularRecommender extends AbstractRecommender {

	private int[] populars;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		populars = new int[numberOfItems];
	}

	@Override
	protected void doPractice() {
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			populars[itemIndex] = trainMatrix.getColumnScope(itemIndex);
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int itemIndex = dicreteFeatures[itemDimension];
		return populars[itemIndex];
	}

}
