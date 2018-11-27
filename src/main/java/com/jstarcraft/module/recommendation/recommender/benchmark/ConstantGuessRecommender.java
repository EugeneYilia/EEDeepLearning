package com.jstarcraft.module.recommendation.recommender.benchmark;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.model.ModelDefinition;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.AbstractRecommender;

/**
 * 常量分数推荐器
 * 
 * @author Birdy
 *
 */
@ModelDefinition(value = { "constant" })
public class ConstantGuessRecommender extends AbstractRecommender {

	private float constant;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		// 默认使用最高最低分的平均值
		constant = (minimumOfScore + maximumOfScore) / 2F;
		// TODO 支持配置分数
		constant = configuration.getFloat("recommend.constant-guess.score", constant);
	}

	@Override
	protected void doPractice() {
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		return constant;
	}

}
