package com.jstarcraft.module.recommendation.task;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.evaluator.Evaluator;
import com.jstarcraft.module.recommendation.evaluator.rating.MAEEvaluator;
import com.jstarcraft.module.recommendation.evaluator.rating.MPEEvaluator;
import com.jstarcraft.module.recommendation.evaluator.rating.MSEEvaluator;
import com.jstarcraft.module.recommendation.recommender.Recommender;

/**
 * RecommenderJob
 *
 * @author WangYuFeng
 */
// TODO 核心目标是解析配置,装配转换器/分割器/推荐器/评估器,完成指定的评分预测或者排序预测任务.
public class RatingTask extends AbstractTask {

	public RatingTask(Configuration configuration) {
		super(configuration);
	}

	@Override
	protected Collection<Evaluator> getEvaluators(SparseMatrix featureMatrix) {
		Collection<Evaluator> evaluators = new LinkedList<>();
		evaluators.add(new MAEEvaluator());
		evaluators.add(new MPEEvaluator(0.01F));
		evaluators.add(new MSEEvaluator());
		return evaluators;
	}

	@Override
	protected Collection<Float> check(int userIndex) {
		int from = testPaginations[userIndex], to = testPaginations[userIndex + 1];
		List<Float> scoreList = new ArrayList<>(to - from);
		for (int index = from, size = to; index < size; index++) {
			int position = testPositions[index];
			scoreList.add(testMarker.getMark(position));
		}
		return scoreList;
	}

	@Override
	protected List<KeyValue<Integer, Float>> recommend(Recommender recommender, int userIndex) {
		int from = testPaginations[userIndex], to = testPaginations[userIndex + 1];
		int[] discreteFeatures = new int[testMarker.getDiscreteOrder()];
		float[] continuousFeatures = new float[testMarker.getContinuousOrder()];
		List<KeyValue<Integer, Float>> recommendList = new ArrayList<>(to - from);
		for (int index = from, size = to; index < size; index++) {
			int position = testPositions[index];
			for (int dimension = 0; dimension < testMarker.getDiscreteOrder(); dimension++) {
				discreteFeatures[dimension] = testMarker.getDiscreteFeature(dimension, position);
			}
			for (int dimension = 0; dimension < testMarker.getContinuousOrder(); dimension++) {
				continuousFeatures[dimension] = testMarker.getContinuousFeature(dimension, position);
			}
			recommendList.add(new KeyValue<>(discreteFeatures[itemDimension], recommender.predict(discreteFeatures, continuousFeatures)));
		}
		return recommendList;
	}

}
