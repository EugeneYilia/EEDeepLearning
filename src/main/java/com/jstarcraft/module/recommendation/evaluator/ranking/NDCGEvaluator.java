package com.jstarcraft.module.recommendation.evaluator.ranking;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.recommendation.evaluator.RankingEvaluator;

/**
 * <pre>
 * NDCG = Normalized Discounted Cumulative Gain
 * https://en.wikipedia.org/wiki/Discounted_cumulative_gain
 * </pre>
 *
 * @author Birdy
 */
public class NDCGEvaluator extends RankingEvaluator {

	private List<Float> idcgs;

	public NDCGEvaluator(int size) {
		super(size);
		idcgs = new ArrayList<>(size + 1);
		idcgs.add(0F);
		for (int index = 0; index < size; index++) {
			idcgs.add((float) (1F / MathUtility.logarithm(index + 2F, 2) + idcgs.get(index)));
		}
	}

	@Override
	protected float measure(Collection<Integer> checkCollection, List<KeyValue<Integer, Float>> recommendList) {
		if (recommendList.size() > size) {
			recommendList = recommendList.subList(0, size);
		}
		float dcg = 0F;
		// calculate DCG
		int size = recommendList.size();
		for (int index = 0; index < size; index++) {
			int itemIndex = recommendList.get(index).getKey();
			if (!checkCollection.contains(itemIndex)) {
				continue;
			}
			dcg += (float) (1F / MathUtility.logarithm(index + 2F, 2));
		}
		return dcg / idcgs.get(checkCollection.size() < size ? checkCollection.size() : size);
	}

}
