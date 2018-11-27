package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.Probability;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.math.structure.tensor.SparseTensor;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * Jahrer and Toscher, Collaborative Filtering Ensemble for Ranking, JMLR, 2012
 * (KDD Cup 2011 Track 2).
 *
 * @author guoguibing and Keqiang Wang
 */
public class RankSGDRecommender extends MatrixFactorizationRecommender {
	// item sampling probabilities sorted ascendingly

	protected Probability itemProbabilities;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		// compute item sampling probability
		itemProbabilities = new Probability(numberOfItems, (index, value, message) -> {
			int userSize = trainMatrix.getColumnScope(index);
			// sample items based on popularity
			return (userSize + 0F) / numberOfActions;
		});
	}

	@Override
	protected void doPractice() {
		List<Set<Integer>> userItemSet = getUserItemSet(trainMatrix);
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			// for each rated user-item (u,i) pair
			for (MatrixScalar term : trainMatrix) {
				int userIndex = term.getRow();
				Set<Integer> itemSet = userItemSet.get(userIndex);
				int positiveItemIndex = term.getColumn();
				float positiveRate = term.getValue();
				int negativeItemIndex = -1;

				do {
					// draw an item j with probability proportional to
					// popularity
					negativeItemIndex = itemProbabilities.random();
					// ensure that it is unrated by user u
				} while (itemSet.contains(negativeItemIndex));

				float negativeRate = 0F;
				// compute predictions
				float error = (predict(userIndex, positiveItemIndex) - predict(userIndex, negativeItemIndex)) - (positiveRate - negativeRate);
				totalLoss += error * error;

				// update vectors
				float value = learnRate * error;
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactor = userFactors.getValue(userIndex, factorIndex);
					float positiveItemFactor = itemFactors.getValue(positiveItemIndex, factorIndex);
					float negativeItemFactor = itemFactors.getValue(negativeItemIndex, factorIndex);

					userFactors.shiftValue(userIndex, factorIndex, -value * (positiveItemFactor - negativeItemFactor));
					itemFactors.shiftValue(positiveItemIndex, factorIndex, -value * userFactor);
					itemFactors.shiftValue(negativeItemIndex, factorIndex, value * userFactor);
				}
			}

			totalLoss *= 0.5D;
			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			isLearned(iterationStep);
			currentLoss = totalLoss;
		}
	}

	protected List<Set<Integer>> getUserItemSet(SparseMatrix sparseMatrix) {
		List<Set<Integer>> userItemSet = new ArrayList<>(numberOfUsers);
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			SparseVector userVector = sparseMatrix.getRowVector(userIndex);
			Set<Integer> indexes = new HashSet<>(userVector.getElementSize());
			userVector.getTermKeys(indexes);
			userItemSet.add(indexes);
		}
		return userItemSet;
	}

}
