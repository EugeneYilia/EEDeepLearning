package com.jstarcraft.module.recommendation.recommender.extend;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.tensor.SparseTensor;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.exception.RecommendationException;
import com.jstarcraft.module.recommendation.recommender.AbstractRecommender;

/**
 * Weighted Slope One: Lemire and Maclachlan, <strong> Slope One Predictors for
 * Online Rating-Based Collaborative Filtering </strong>, SDM 2005.
 *
 * @author GuoGuibing and Keqiang Wang
 */
public class SlopeOneRecommender extends AbstractRecommender {
	/**
	 * matrices for item-item differences with number of occurrences/cardinal
	 */
	private DenseMatrix deviationMatrix, cardinalMatrix;

	/**
	 * initialization
	 *
	 * @throws RecommendationException
	 *             if error occurs
	 */
	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		deviationMatrix = DenseMatrix.valueOf(numberOfItems, numberOfItems);
		cardinalMatrix = DenseMatrix.valueOf(numberOfItems, numberOfItems);
	}

	/**
	 * train model
	 *
	 * @throws RecommendationException
	 *             if error occurs
	 */
	@Override
	protected void doPractice() {
		// compute items' differences
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			SparseVector itemVector = trainMatrix.getRowVector(userIndex);
			for (VectorScalar leftTerm : itemVector) {
				float leftRate = leftTerm.getValue();
				for (VectorScalar rightTerm : itemVector) {
					if (leftTerm.getIndex() != rightTerm.getIndex()) {
						float rightRate = rightTerm.getValue();
						deviationMatrix.shiftValue(leftTerm.getIndex(), rightTerm.getIndex(), leftRate - rightRate);
						cardinalMatrix.shiftValue(leftTerm.getIndex(), rightTerm.getIndex(), 1);
					}
				}
			}
		}

		// normalize differences
		deviationMatrix.mapValues((row, column, value, message) -> {
			float cardinal = cardinalMatrix.getValue(row, column);
			return cardinal > 0F ? value / cardinal : value;
		}, null, MathCalculator.PARALLEL);
	}

	/**
	 * predict a specific rating for user userIdx on item itemIdx.
	 *
	 * @param userIndex
	 *            user index
	 * @param itemIndex
	 *            item index
	 * @return predictive rating for user userIdx on item itemIdx
	 * @throws RecommendationException
	 *             if error occurs
	 */
	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		SparseVector userVector = trainMatrix.getRowVector(userIndex);
		float value = 0F, sum = 0F;
		for (VectorScalar term : userVector) {
			if (itemIndex == term.getIndex()) {
				continue;
			}
			double cardinary = cardinalMatrix.getValue(itemIndex, term.getIndex());
			if (cardinary > 0F) {
				value += (deviationMatrix.getValue(itemIndex, term.getIndex()) + term.getValue()) * cardinary;
				sum += cardinary;
			}
		}
		return sum > 0F ? value / sum : meanOfScore;
	}

}
