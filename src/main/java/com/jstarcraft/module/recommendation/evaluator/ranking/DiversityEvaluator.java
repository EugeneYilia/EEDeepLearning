package com.jstarcraft.module.recommendation.evaluator.ranking;

import java.util.Collection;
import java.util.List;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.matrix.SymmetryMatrix;
import com.jstarcraft.module.recommendation.evaluator.RankingEvaluator;

/**
 * 多样性评估器
 *
 * @author Birdy
 */
public class DiversityEvaluator extends RankingEvaluator {

	private SymmetryMatrix similarityMatrix;

	public DiversityEvaluator(int size, SymmetryMatrix similarityMatrix) {
		super(size);
		this.similarityMatrix = similarityMatrix;
	}

	@Override
	protected float measure(Collection<Integer> checkCollection, List<KeyValue<Integer, Float>> recommendList) {
		if (recommendList.size() > size) {
			recommendList = recommendList.subList(0, size);
		}
		float diversity = 0F;
		int size = recommendList.size();
		for (int indexOut = 0; indexOut < size; indexOut++) {
			for (int indexIn = indexOut + 1; indexIn < size; indexIn++) {
				int itemOut = recommendList.get(indexOut).getKey();
				int itemIn = recommendList.get(indexIn).getKey();
				diversity += 1F - similarityMatrix.getValue(itemOut, itemIn);
				diversity += 1F - similarityMatrix.getValue(itemIn, itemOut);
			}
		}
		return diversity / (size * (size - 1));
	}

}
