package com.jstarcraft.module.recommendation.evaluator.ranking;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.recommendation.evaluator.RankingEvaluator;

/**
 * ROC曲线下的面积评估器
 * 
 * <pre>
 * AUC = Area Under roc Curve(ROC曲线下面积)
 * http://www.cnblogs.com/lixiaolun/p/4053499.html
 * </pre>
 *
 * @author Birdy
 */
public class AUCEvaluator extends RankingEvaluator {

	public AUCEvaluator(int size) {
		super(size);
	}

	@Override
	protected float measure(Collection<Integer> checkCollection, List<KeyValue<Integer, Float>> recommendList) {
		// 推荐物品集合(大小不能超过TopN)
		int evaluateSize = recommendList.size();
		if (evaluateSize > size) {
			recommendList = recommendList.subList(0, size);
		}
		int hitCount = 0, missCount = 0;
		Set<Integer> recommendItems = new HashSet<>();
		for (KeyValue<Integer, Float> keyValue : recommendList) {
			recommendItems.add(keyValue.getKey());
			if (checkCollection.contains(keyValue.getKey())) {
				hitCount++;
			} else {
				missCount++;
			}
		}

		int evaluateSum = (checkCollection.size() + evaluateSize - recommendList.size() - hitCount) * hitCount;
		if (evaluateSum == 0) {
			return 0.5F;
		}
		int hitSum = 0;
		hitCount = 0;
		for (Integer itemIndex : checkCollection) {
			if (!recommendItems.contains(itemIndex)) {
				hitSum += hitCount;
			} else {
				hitCount++;
			}
		}
		hitSum += hitCount * (evaluateSize - missCount);
		return (hitSum + 0F) / evaluateSum;
	}
}