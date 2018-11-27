package com.jstarcraft.module.recommendation.evaluator.rate;

import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.recommendation.evaluator.AbstractRatingEvaluatorTestCase;
import com.jstarcraft.module.recommendation.evaluator.Evaluator;
import com.jstarcraft.module.recommendation.evaluator.rating.MSEEvaluator;

public class MSEEvaluatorTestCase extends AbstractRatingEvaluatorTestCase {

	@Override
	protected Evaluator<?> getEvaluator(SparseMatrix featureMatrix) {
		return new MSEEvaluator();
	}

	@Override
	protected float getMeasure() {
		return 376666.24320291774F;
	}

}