package com.jstarcraft.module.recommendation.evaluator.rate;

import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.recommendation.evaluator.AbstractRatingEvaluatorTestCase;
import com.jstarcraft.module.recommendation.evaluator.Evaluator;
import com.jstarcraft.module.recommendation.evaluator.rating.MAEEvaluator;

public class MAEEvaluatorTestCase extends AbstractRatingEvaluatorTestCase {

	@Override
	protected Evaluator<?> getEvaluator(SparseMatrix featureMatrix) {
		return new MAEEvaluator();
	}

	@Override
	protected float getMeasure() {
		return 546.6342838196286F;
	}

}