package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import java.util.Map.Entry;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * Gedikli et al., <strong>RF-Rec: Fast and Accurate Computation of
 * Recommendations based on Rating Frequencies</strong>, IEEE (CEC) 2011,
 * Luxembourg, 2011, pp. 50-57. <br>
 * <p>
 * <strong>Remark:</strong> This implementation does not support half-star
 * ratings.
 * 
 * @author bin wu(Email:wubin@gs.zzu.edu.cn)
 */
public class RFRecRecommender extends MatrixFactorizationRecommender {
	/**
	 * The average ratings of users
	 */
	private DenseVector userMeans;

	/**
	 * The average ratings of items
	 */
	private DenseVector itemMeans;

	/**
	 * The number of ratings per rating value per user
	 */
	private DenseMatrix userRateFrequencies;

	/**
	 * The number of ratings per rating value per item
	 */
	private DenseMatrix itemRateFrequencies;

	/**
	 * User weights learned by the gradient solver
	 */
	private DenseVector userWeights;

	/**
	 * Item weights learned by the gradient solver.
	 */
	private DenseVector itemWeights;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		// Calculate the average ratings
		userMeans = DenseVector.valueOf(numberOfUsers);
		itemMeans = DenseVector.valueOf(numberOfItems);
		userWeights = DenseVector.valueOf(numberOfUsers);
		itemWeights = DenseVector.valueOf(numberOfItems);

		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			SparseVector userVector = trainMatrix.getRowVector(userIndex);
			userMeans.setValue(userIndex, userVector.getSum(false) / userVector.getElementSize());
			userWeights.setValue(userIndex, 0.6F + RandomUtility.randomFloat(0.01F));
		}
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			SparseVector itemVector = trainMatrix.getColumnVector(itemIndex);
			itemMeans.setValue(itemIndex, itemVector.getSum(false) / itemVector.getElementSize());
			itemWeights.setValue(itemIndex, 0.4F + RandomUtility.randomFloat(0.01F));
		}
		// Calculate the frequencies.
		// Users,items
		userRateFrequencies = DenseMatrix.valueOf(numberOfUsers, numberOfActions);
		itemRateFrequencies = DenseMatrix.valueOf(numberOfItems, numberOfActions);
		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			int rateIndex = scoreIndexes.get(term.getValue());
			userRateFrequencies.shiftValue(userIndex, rateIndex, 1F);
			itemRateFrequencies.shiftValue(itemIndex, rateIndex, 1F);
		}
	}

	@Override
	protected void doPractice() {
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			for (MatrixScalar term : trainMatrix) {
				int userIndex = term.getRow();
				int itemIndex = term.getColumn();
				float error = term.getValue() - predict(userIndex, itemIndex);

				// Gradient-Step on user weights.
				float userWeight = userWeights.getValue(userIndex) + learnRate * (error - userRegularization * userWeights.getValue(userIndex));
				userWeights.setValue(userIndex, userWeight);

				// Gradient-Step on item weights.
				float itemWeight = itemWeights.getValue(itemIndex) + learnRate * (error - itemRegularization * itemWeights.getValue(itemIndex));
				itemWeights.setValue(itemIndex, itemWeight);
			}
		}
	}

	/**
	 * Returns 1 if the rating is similar to the rounded average value
	 *
	 * @param mean
	 *            the average
	 * @param rate
	 *            the rating
	 * @return 1 when the values are equal
	 */
	private int isMean(double mean, int rate) {
		return Math.round(mean) == rate ? 1 : 0;
	}

	@Override
	protected float predict(int userIndex, int itemIndex) {
		float value = meanOfScore;
		float userSum = userRateFrequencies.getRowVector(userIndex).getSum(false);
		float itemSum = itemRateFrequencies.getRowVector(itemIndex).getSum(false);
		float userMean = userMeans.getValue(userIndex);
		float itemMean = itemMeans.getValue(itemIndex);

		if (userSum > 0F && itemSum > 0F && userMean > 0F && itemMean > 0F) {
			float numeratorUser = 0F;
			float denominatorUser = 0F;
			float numeratorItem = 0F;
			float denominatorItem = 0F;
			float frequency = 0F;
			// Go through all the possible rating values
			for (Entry<Float, Integer> term : scoreIndexes.entrySet()) {
				int rateIndex = term.getValue();
				// user component
				frequency = userRateFrequencies.getValue(userIndex, rateIndex);
				frequency = frequency + 1 + isMean(userMean, rateIndex);
				numeratorUser += frequency * rateIndex;
				denominatorUser += frequency;

				// item component
				frequency = itemRateFrequencies.getValue(itemIndex, rateIndex);
				frequency = frequency + 1 + isMean(itemMean, rateIndex);
				numeratorItem += frequency * rateIndex;
				denominatorItem += frequency;
			}

			float userWeight = userWeights.getValue(userIndex);
			float itemWeight = itemWeights.getValue(itemIndex);
			value = userWeight * numeratorUser / denominatorUser + itemWeight * numeratorItem / denominatorItem;
		} else {
			// if the user or item weren't known in the training phase...
			if (userSum == 0F || userMean == 0F) {
				if (itemMean != 0F) {
					return itemMean;
				} else {
					return meanOfScore;
				}
			}
			if (itemSum == 0F || itemMean == 0F) {
				if (userMean != 0F) {
					return userMean;
				} else {
					// Some heuristic -> a bit above the average rating
					return meanOfScore;
				}
			}
		}
		return value;
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		return predict(userIndex, itemIndex);
	}

}
