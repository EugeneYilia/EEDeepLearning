package com.jstarcraft.module.recommendation.evaluator.rank;

import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.recommendation.evaluator.AbstractRankingEvaluatorTestCase;
import com.jstarcraft.module.recommendation.evaluator.Evaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.RecallEvaluator;

public class RecallEvaluatorTestCase extends AbstractRankingEvaluatorTestCase {

	@Override
	protected Evaluator<?> getEvaluator(SparseMatrix featureMatrix) {
		return new RecallEvaluator(10);
	}

	@Override
	protected float getMeasure() {
		return 0.6744031830238727F;
	}

}
