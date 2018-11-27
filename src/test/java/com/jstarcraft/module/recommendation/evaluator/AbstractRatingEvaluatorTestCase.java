package com.jstarcraft.module.recommendation.evaluator;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.recommendation.recommender.Recommender;

public abstract class AbstractRatingEvaluatorTestCase extends AbstractEvaluatorTestCase<Float> {

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
