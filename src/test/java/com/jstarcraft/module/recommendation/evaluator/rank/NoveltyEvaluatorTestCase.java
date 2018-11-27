package com.jstarcraft.module.recommendation.evaluator.rank;

import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.recommendation.evaluator.AbstractRankingEvaluatorTestCase;
import com.jstarcraft.module.recommendation.evaluator.Evaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.NoveltyEvaluator;

public class NoveltyEvaluatorTestCase extends AbstractRankingEvaluatorTestCase {

	@Override
	protected Evaluator<?> getEvaluator(SparseMatrix featureMatrix) {
		return new NoveltyEvaluator(10, featureMatrix);
	}

	@Override
	protected float getMeasure() {
		return 16.829857137294585F;
	}

}