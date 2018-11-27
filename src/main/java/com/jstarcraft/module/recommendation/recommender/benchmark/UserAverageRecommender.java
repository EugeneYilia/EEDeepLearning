package com.jstarcraft.module.recommendation.recommender.benchmark;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.tensor.SparseTensor;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.AbstractRecommender;

/**
 * 用户平均分数推荐器
 * 
 * @author Birdy
 *
 */
@ModelDefinition(value = { "userDimension", "userMeans" })
public class UserAverageRecommender extends AbstractRecommender {

	/** 用户平均分数 */
	private float[] userMeans;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		userMeans = new float[numberOfUsers];
	}

	@Override
	protected void doPractice() {
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			SparseVector userVector = trainMatrix.getRowVector(userIndex);
			userMeans[userIndex] = userVector.getElementSize() == 0 ? meanOfScore : userVector.getSum(false) / userVector.getElementSize();
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		return userMeans[userIndex];
	}

}
