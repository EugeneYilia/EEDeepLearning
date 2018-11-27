package com.jstarcraft.module.recommendation.recommender;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.recommendation.configure.Configuration;

/**
 * 推荐器
 * 
 * <pre>
 * 注意区分每个阶段的职责:
 * 准备阶段关注数据,负责根据算法转换数据;
 * 训练阶段关注参数,负责根据参数获得模型;
 * 预测阶段关注模型,负责根据模型预测得分;
 * </pre>
 * 
 * @author Birdy
 *
 */
public interface Recommender {

	/**
	 * 准备
	 * 
	 * @param configuration
	 */
	void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space);
	// void prepare(Configuration configuration, SparseTensor trainTensor,
	// SparseTensor testTensor, DataSpace storage);

	/**
	 * 训练
	 * 
	 * @param trainTensor
	 * @param testTensor
	 * @param contextModels
	 */
	void practice();

	/**
	 * 预测
	 * 
	 * @param userIndex
	 * @param itemIndex
	 * @param featureIndexes
	 * @return
	 */
	float predict(int[] dicreteFeatures, float[] continuousFeatures);
	// double predict(int userIndex, int itemIndex, int... featureIndexes);

}
