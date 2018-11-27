package com.jstarcraft.module.recommendation.evaluator.rank;

import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.math.structure.matrix.SymmetryMatrix;
import com.jstarcraft.module.recommendation.evaluator.AbstractRankingEvaluatorTestCase;
import com.jstarcraft.module.recommendation.evaluator.Evaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.DiversityEvaluator;
import com.jstarcraft.module.similarity.CosineSimilarity;
import com.jstarcraft.module.similarity.Similarity;

public class DiversityEvaluatorTestCase extends AbstractRankingEvaluatorTestCase {

	@Override
	protected Evaluator<?> getEvaluator(SparseMatrix featureMatrix) {
		// Item Similarity Matrix
		Similarity similarity = new CosineSimilarity();
		SymmetryMatrix similarityMatrix = similarity.makeSimilarityMatrix(featureMatrix, true, 0F);
		return new DiversityEvaluator(10, similarityMatrix);
	}

	@Override
	protected float getMeasure() {
		return 0.05495300284160192F;
	}

}