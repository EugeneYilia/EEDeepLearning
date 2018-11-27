package com.jstarcraft.module.recommendation.recommender.extend;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.exception.RecommendationException;
import com.jstarcraft.module.recommendation.recommender.AbstractRecommender;

/**
 * Related Work:
 * <ul>
 * <li><a href=
 * "http://www.cs.carleton.edu/cs_comps/0607/recommend/recommender/pd.html">A
 * brief introduction to Personality Diagnosis</a></li>
 * </ul>
 *
 * @author guoguibing and Keqiang Wang
 */
public class PersonalityDiagnosisRecommender extends AbstractRecommender {
	/**
	 * Gaussian noise: 2.5 suggested in the paper
	 */
	private float sigma;

	/**
	 * prior probability
	 */
	private float prior;

	private ArrayList<Float> values;

	/**
	 * initialization
	 *
	 * @throws RecommendationException
	 *             if error occurs
	 */
	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		prior = 1F / numberOfUsers;
		sigma = configuration.getFloat("rec.PersonalityDiagnosis.sigma");
		values = new ArrayList<>(scoreIndexes.keySet());
	}

	@Override
	protected void doPractice() {
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
		float[] probabilities = new float[scoreIndexes.size()];
		SparseVector itemVector = trainMatrix.getColumnVector(itemIndex);
		SparseVector rightUserVector = trainMatrix.getRowVector(userIndex);
		for (VectorScalar term : itemVector) {
			// other users who rated item j
			userIndex = term.getIndex();
			float rate = term.getValue();
			float probability = 1F;
			SparseVector leftUserVector = trainMatrix.getRowVector(userIndex);
			int leftIndex = 0, rightIndex = 0, leftSize = leftUserVector.getElementSize(), rightSize = rightUserVector.getElementSize();
			if (leftSize != 0 && rightSize != 0) {
				Iterator<VectorScalar> leftIterator = leftUserVector.iterator();
				Iterator<VectorScalar> rightIterator = rightUserVector.iterator();
				VectorScalar leftTerm = leftIterator.next();
				VectorScalar rightTerm = rightIterator.next();
				// 判断两个有序数组中是否存在相同的数字
				while (leftIndex < leftSize && rightIndex < rightSize) {
					if (leftTerm.getIndex() == rightTerm.getIndex()) {
						probability *= MathUtility.gaussian(rightTerm.getValue(), leftTerm.getValue(), sigma);
						if (leftIterator.hasNext()) {
							leftTerm = leftIterator.next();
						}
						if (rightIterator.hasNext()) {
							rightTerm = rightIterator.next();
						}
						leftIndex++;
						rightIndex++;
					} else if (leftTerm.getIndex() > rightTerm.getIndex()) {
						if (rightIterator.hasNext()) {
							rightTerm = rightIterator.next();
						}
						rightIndex++;
					} else if (leftTerm.getIndex() < rightTerm.getIndex()) {
						if (leftIterator.hasNext()) {
							leftTerm = leftIterator.next();
						}
						leftIndex++;
					}
				}
			}
			for (Entry<Float, Integer> entry : scoreIndexes.entrySet()) {
				probabilities[entry.getValue()] += MathUtility.gaussian(entry.getKey(), rate, sigma) * probability;
			}
		}
		for (Entry<Float, Integer> entry : scoreIndexes.entrySet()) {
			probabilities[entry.getValue()] *= prior;
		}
		int valueIndex = 0;
		float probability = Float.MIN_VALUE;
		for (int rateIndex = 0; rateIndex < probabilities.length; rateIndex++) {
			if (probabilities[rateIndex] > probability) {
				probability = probabilities[rateIndex];
				valueIndex = rateIndex;
			}
		}
		return values.get(valueIndex);
	}

}
