package com.jstarcraft.module.recommendation.evaluator.rank;

import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.recommendation.evaluator.AbstractRankingEvaluatorTestCase;
import com.jstarcraft.module.recommendation.evaluator.Evaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.PrecisionEvaluator;

public class PrecisionEvaluatorTestCase extends AbstractRankingEvaluatorTestCase {

	@Override
	protected Evaluator<?> getEvaluator(SparseMatrix featureMatrix) {
		return new PrecisionEvaluator(10);
	}

	@Override
	protected float getMeasure() {
		return 0.06744031830238627F;
	}

}