package com.jstarcraft.module.recommendation.recommender.benchmark;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.AbstractRecommender;

/**
 * 物品平均分数推荐器
 * 
 * @author Birdy
 *
 */
@ModelDefinition(value = { "itemDimension", "itemMeans" })
public class ItemAverageRecommender extends AbstractRecommender {

	/** 物品平均分数 */
	private float[] itemMeans;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		itemMeans = new float[numberOfItems];
	}

	@Override
	protected void doPractice() {
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			SparseVector itemVector = trainMatrix.getColumnVector(itemIndex);
			itemMeans[itemIndex] = itemVector.getElementSize() == 0 ? meanOfScore : itemVector.getSum(false) / itemVector.getElementSize();
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int itemIndex = dicreteFeatures[itemDimension];
		return itemMeans[itemIndex];
	}

}
