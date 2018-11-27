package com.jstarcraft.module.recommendation.recommender.context.ranking;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.SocialRecommender;

/**
 * Social Bayesian Personalized Ranking (SBPR)
 * <p>
 * Zhao et al., <strong>Leveraging Social Connections to Improve Personalized
 * Ranking for Collaborative Filtering</strong>, CIKM 2014.
 *
 * @author guoguibing and Keqiang Wang
 */
// TODO 仍需重构
public class SBPRRecommender extends SocialRecommender {
	/**
	 * items biases vector
	 */
	private DenseVector itemBiases;

	/**
	 * bias regularization
	 */
	protected float regBias;

	/**
	 * find items rated by trusted neighbors only
	 */
	// TODO 考虑重构为List<Set<Integer>>
	private List<List<Integer>> socialItemList;

	// TODO 考虑重构到抽象类
	private List<Set<Integer>> userItemSet;

	private List<Set<Integer>> getUserItemSet(SparseMatrix sparseMatrix) {
		List<Set<Integer>> userItemSet = new ArrayList<>(numberOfUsers);
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			SparseVector userVector = sparseMatrix.getRowVector(userIndex);
			Set<Integer> indexes = new HashSet<>(userVector.getElementSize());
			userVector.getTermKeys(indexes);
			userItemSet.add(indexes);
		}
		return userItemSet;
	}

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		regBias = configuration.getFloat("rec.bias.regularization", 0.01F);
		// cacheSpec = conf.get("guava.cache.spec",
		// "maximumSize=5000,expireAfterAccess=50m");

		itemBiases = DenseVector.valueOf(numberOfItems, VectorMapper.RANDOM);

		userItemSet = getUserItemSet(trainMatrix);

		// TODO 考虑重构
		// find items rated by trusted neighbors only
		socialItemList = new ArrayList<>(numberOfUsers);

		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			SparseVector userVector = trainMatrix.getRowVector(userIndex);
			Set<Integer> itemSet = userItemSet.get(userIndex);
			// find items rated by trusted neighbors only

			SparseVector socialVector = socialMatrix.getRowVector(userIndex);
			List<Integer> socialList = new LinkedList<>();
			for (VectorScalar term : socialVector) {
				int socialIndex = term.getIndex();
				userVector = trainMatrix.getRowVector(socialIndex);
				for (VectorScalar enrty : userVector) {
					int itemIndex = enrty.getIndex();
					// v's rated items
					if (!itemSet.contains(itemIndex) && !socialList.contains(itemIndex)) {
						socialList.add(itemIndex);
					}
				}
			}
			socialItemList.add(new ArrayList<>(socialList));
		}
	}

	@Override
	protected void doPractice() {
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			for (int sampleIndex = 0, sampleTimes = numberOfUsers * 100; sampleIndex < sampleTimes; sampleIndex++) {
				// uniformly draw (userIdx, posItemIdx, k, negItemIdx)
				int userIndex, positiveItemIndex, negativeItemIndex;
				// userIdx
				SparseVector userVector;
				do {
					userIndex = RandomUtility.randomInteger(numberOfUsers);
					userVector = trainMatrix.getRowVector(userIndex);
				} while (userVector.getElementSize() == 0);

				// positive item index
				positiveItemIndex = userVector.randomKey();
				float positiveRate = predict(userIndex, positiveItemIndex);

				// social Items List
				// TODO 应该修改为Set<Integer>合适点.
				List<Integer> socialList = socialItemList.get(userIndex);
				Set<Integer> itemSet = userItemSet.get(userIndex);
				do {
					negativeItemIndex = RandomUtility.randomInteger(numberOfItems);
				} while (itemSet.contains(negativeItemIndex) || socialList.contains(negativeItemIndex));
				float negativeRate = predict(userIndex, negativeItemIndex);

				if (socialList.size() > 0) {
					// if having social neighbors
					int itemIndex = socialList.get(RandomUtility.randomInteger(socialList.size()));
					float socialRate = predict(userIndex, itemIndex);
					SparseVector socialVector = socialMatrix.getRowVector(userIndex);
					float socialWeight = 0F;
					for (VectorScalar term : socialVector) {
						int socialIndex = term.getIndex();
						itemSet = userItemSet.get(socialIndex);
						if (itemSet.contains(itemIndex)) {
							socialWeight += 1;
						}
					}
					float positiveError = (positiveRate - socialRate) / (1 + socialWeight);
					float negativeError = socialRate - negativeRate;
					float positiveGradient = MathUtility.logistic(-positiveError), negativeGradient = MathUtility.logistic(-negativeError);
					float error = (float) (-Math.log(1 - positiveGradient) - Math.log(1 - negativeGradient));
					totalLoss += error;

					// update bi, bk, bj
					float positiveBias = itemBiases.getValue(positiveItemIndex);
					itemBiases.shiftValue(positiveItemIndex, learnRate * (positiveGradient / (1F + socialWeight) - regBias * positiveBias));
					totalLoss += regBias * positiveBias * positiveBias;
					float socialBias = itemBiases.getValue(itemIndex);
					itemBiases.shiftValue(itemIndex, learnRate * (-positiveGradient / (1F + socialWeight) + negativeGradient - regBias * socialBias));
					totalLoss += regBias * socialBias * socialBias;
					float negativeBias = itemBiases.getValue(negativeItemIndex);
					itemBiases.shiftValue(negativeItemIndex, learnRate * (-negativeGradient - regBias * negativeBias));
					totalLoss += regBias * negativeBias * negativeBias;

					// update P, Q
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						float userFactor = userFactors.getValue(userIndex, factorIndex);
						float positiveFactor = itemFactors.getValue(positiveItemIndex, factorIndex);
						float itemFactor = itemFactors.getValue(itemIndex, factorIndex);
						float negativeFactor = itemFactors.getValue(negativeItemIndex, factorIndex);
						float delta = positiveGradient * (positiveFactor - itemFactor) / (1F + socialWeight) + negativeGradient * (itemFactor - negativeFactor);
						userFactors.shiftValue(userIndex, factorIndex, learnRate * (delta - userRegularization * userFactor));
						itemFactors.shiftValue(positiveItemIndex, factorIndex, learnRate * (positiveGradient * userFactor / (1F + socialWeight) - itemRegularization * positiveFactor));
						itemFactors.shiftValue(negativeItemIndex, factorIndex, learnRate * (negativeGradient * (-userFactor) - itemRegularization * negativeFactor));
						delta = positiveGradient * (-userFactor / (1F + socialWeight)) + negativeGradient * userFactor;
						itemFactors.shiftValue(itemIndex, factorIndex, learnRate * (delta - itemRegularization * itemFactor));
						totalLoss += userRegularization * userFactor * userFactor + itemRegularization * positiveFactor * positiveFactor + itemRegularization * negativeFactor * negativeFactor + itemRegularization * itemFactor * itemFactor;
					}
				} else {
					// if no social neighbors, the same as BPR
					float error = positiveRate - negativeRate;
					totalLoss += error;
					float gradient = MathUtility.logistic(-error);

					// update bi, bj
					float positiveBias = itemBiases.getValue(positiveItemIndex);
					itemBiases.shiftValue(positiveItemIndex, learnRate * (gradient - regBias * positiveBias));
					totalLoss += regBias * positiveBias * positiveBias;
					float negativeBias = itemBiases.getValue(negativeItemIndex);
					itemBiases.shiftValue(negativeItemIndex, learnRate * (-gradient - regBias * negativeBias));
					totalLoss += regBias * negativeBias * negativeBias;

					// update user factors, item factors
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						float userFactor = userFactors.getValue(userIndex, factorIndex);
						float positiveFactor = itemFactors.getValue(positiveItemIndex, factorIndex);
						float negItemFactorValue = itemFactors.getValue(negativeItemIndex, factorIndex);
						userFactors.shiftValue(userIndex, factorIndex, learnRate * (gradient * (positiveFactor - negItemFactorValue) - userRegularization * userFactor));
						itemFactors.shiftValue(positiveItemIndex, factorIndex, learnRate * (gradient * userFactor - itemRegularization * positiveFactor));
						itemFactors.shiftValue(negativeItemIndex, factorIndex, learnRate * (gradient * (-userFactor) - itemRegularization * negItemFactorValue));
						totalLoss += userRegularization * userFactor * userFactor + itemRegularization * positiveFactor * positiveFactor + itemRegularization * negItemFactorValue * negItemFactorValue;
					}
				}
			}

			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			isLearned(iterationStep);
			currentLoss = totalLoss;
		}
	}

	@Override
	protected float predict(int userIndex, int itemIndex) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		DenseVector userVector = userFactors.getRowVector(userIndex);
		DenseVector itemVector = itemFactors.getRowVector(itemIndex);
		return itemBiases.getValue(itemIndex) + scalar.dotProduct(userVector, itemVector).getValue();
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		return predict(userIndex, itemIndex);
	}

}
