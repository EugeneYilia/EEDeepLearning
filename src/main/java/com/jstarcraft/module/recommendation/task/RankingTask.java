package com.jstarcraft.module.recommendation.task;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.evaluator.Evaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.AUCEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.MAPEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.MRREvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.NDCGEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.NoveltyEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.PrecisionEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.RecallEvaluator;
import com.jstarcraft.module.recommendation.recommender.Recommender;

/**
 * RecommenderJob
 *
 * @author WangYuFeng
 */
// TODO 核心目标是解析配置,装配转换器/分割器/推荐器/评估器,完成指定的评分预测或者排序预测任务.
public class RankingTask extends AbstractTask {

	public RankingTask(Configuration configuration) {
		super(configuration);
	}

	@Override
	protected Collection<Evaluator> getEvaluators(SparseMatrix featureMatrix) {
		Collection<Evaluator> evaluators = new LinkedList<>();
		int size = configuration.getInteger("rec.recommender.ranking.topn", 10);
		evaluators.add(new AUCEvaluator(size));
		evaluators.add(new MAPEvaluator(size));
		evaluators.add(new MRREvaluator(size));
		evaluators.add(new NDCGEvaluator(size));
		evaluators.add(new NoveltyEvaluator(size, featureMatrix));
		evaluators.add(new PrecisionEvaluator(size));
		evaluators.add(new RecallEvaluator(size));
		return evaluators;
	}

	@Override
	protected Collection<Integer> check(int userIndex) {
		Set<Integer> itemSet = new LinkedHashSet<>();
		int from = testPaginations[userIndex], to = testPaginations[userIndex + 1];
		for (int index = from, size = to; index < size; index++) {
			int position = testPositions[index];
			itemSet.add(testMarker.getDiscreteFeature(itemDimension, position));
		}
		return itemSet;
	}

	@Override
	protected List<KeyValue<Integer, Float>> recommend(Recommender recommender, int userIndex) {
		Set<Integer> itemSet = new HashSet<>();
		int from = trainPaginations[userIndex], to = trainPaginations[userIndex + 1];
		for (int index = from, size = to; index < size; index++) {
			int position = trainPositions[index];
			itemSet.add(trainMarker.getDiscreteFeature(itemDimension, position));
		}
		int[] discreteFeatures = new int[trainMarker.getDiscreteOrder()];
		float[] continuousFeatures = new float[trainMarker.getContinuousOrder()];
		if (from < to) {
			int position = trainPositions[to - 1];
			for (int dimension = 0, size = trainMarker.getDiscreteOrder(); dimension < size; dimension++) {
				discreteFeatures[dimension] = trainMarker.getDiscreteFeature(dimension, position);
			}
			for (int dimension = 0, size = trainMarker.getContinuousOrder(); dimension < size; dimension++) {
				continuousFeatures[dimension] = trainMarker.getContinuousFeature(dimension, position);
			}
		}
		discreteFeatures[userDimension] = userIndex;
		List<KeyValue<Integer, Float>> recommendList = new ArrayList<>(numberOfItems - itemSet.size());
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			if (itemSet.contains(itemIndex)) {
				continue;
			}
			discreteFeatures[itemDimension] = itemIndex;
			recommendList.add(new KeyValue<>(itemIndex, recommender.predict(discreteFeatures, continuousFeatures)));
		}
		Collections.sort(recommendList, (left, right) -> {
			return right.getValue().compareTo(left.getValue());
		});
		return recommendList;
	}

}
