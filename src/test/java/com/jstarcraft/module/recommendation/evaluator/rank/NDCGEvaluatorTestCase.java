package com.jstarcraft.module.recommendation.evaluator.rank;

import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.recommendation.evaluator.AbstractRankingEvaluatorTestCase;
import com.jstarcraft.module.recommendation.evaluator.Evaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.NDCGEvaluator;

public class NDCGEvaluatorTestCase extends AbstractRankingEvaluatorTestCase {

	@Override
	protected Evaluator<?> getEvaluator(SparseMatrix featureMatrix) {
		return new NDCGEvaluator(10);
	}

	@Override
	protected float getMeasure() {
		return 0.4448149712043262F;
	}

}
