package com.jstarcraft.module.recommendation.evaluator.rate;

import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.recommendation.evaluator.AbstractRatingEvaluatorTestCase;
import com.jstarcraft.module.recommendation.evaluator.Evaluator;
import com.jstarcraft.module.recommendation.evaluator.rating.MPEEvaluator;

public class MPEEvaluatorTestCase extends AbstractRatingEvaluatorTestCase {

	@Override
	protected Evaluator<?> getEvaluator(SparseMatrix featureMatrix) {
		return new MPEEvaluator(0.01F);
	}

	@Override
	protected float getMeasure() {
		return 0.993368700265252F;
	}

}