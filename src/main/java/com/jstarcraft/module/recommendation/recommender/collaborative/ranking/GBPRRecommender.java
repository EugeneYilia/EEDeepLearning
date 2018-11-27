package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.HashSet;
import java.util.Set;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * Pan and Chen, <strong>GBPR: Group Preference Based Bayesian Personalized
 * Ranking for One-Class Collaborative Filtering</strong>, IJCAI 2013.
 *
 * @author guoguibing and Keqiang Wang
 */
public class GBPRRecommender extends MatrixFactorizationRecommender {
	// TODO rho是什么意思?
	private float rho;
	// TODO gLen是什么意思?
	private int gLen;

	/**
	 * bias regularization
	 */
	private float regBias;

	/**
	 * items biases vector
	 */
	private DenseVector itemBiases;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		itemBiases = DenseVector.valueOf(numberOfItems, VectorMapper.RANDOM);

		rho = configuration.getFloat("rec.gpbr.rho", 1.5f);
		gLen = configuration.getInteger("rec.gpbr.gsize", 2);
	}

	@Override
	protected void doPractice() {
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			// TODO 考虑重构
			DenseMatrix userDeltas = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
			DenseMatrix itemDeltas = DenseMatrix.valueOf(numberOfItems, numberOfFactors);

			for (int sampleIndex = 0, sampleTimes = numberOfUsers * 100; sampleIndex < sampleTimes; sampleIndex++) {
				int userIndex, positiveItemIndex, negativeItemIndex;
				SparseVector userVector;
				do {
					userIndex = RandomUtility.randomInteger(numberOfUsers);
					userVector = trainMatrix.getRowVector(userIndex);
				} while (userVector.getElementSize() == 0);
				positiveItemIndex = userVector.randomKey();

				// users group Set
				Set<Integer> memberSet = new HashSet<>();
				SparseVector positiveItemVector = trainMatrix.getColumnVector(positiveItemIndex);
				if (positiveItemVector.getElementSize() <= gLen) {
					for (VectorScalar entry : positiveItemVector) {
						memberSet.add(entry.getIndex());
					}
				} else {
					memberSet.add(userIndex); // u in G
					while (memberSet.size() < gLen) {
						memberSet.add(positiveItemVector.randomKey());
					}
				}
				float positiveRate = predict(userIndex, positiveItemIndex, memberSet);
				negativeItemIndex = RandomUtility.randomInteger(numberOfItems - userVector.getElementSize());
				for (VectorScalar term : userVector) {
					if (negativeItemIndex >= term.getIndex()) {
						negativeItemIndex++;
					} else {
						break;
					}
				}
				float negativeRate = predict(userIndex, negativeItemIndex);
				float error = positiveRate - negativeRate;
				float value = (float) -Math.log(MathUtility.logistic(error));
				totalLoss += value;
				value = MathUtility.logistic(-error);

				// update bi, bj
				float positiveBias = itemBiases.getValue(positiveItemIndex);
				itemBiases.shiftValue(positiveItemIndex, learnRate * (value - regBias * positiveBias));
				float negativeBias = itemBiases.getValue(negativeItemIndex);
				itemBiases.shiftValue(negativeItemIndex, learnRate * (-value - regBias * negativeBias));

				// update Pw
				float averageWeight = 1F / memberSet.size();
				float memberSums[] = new float[numberOfFactors];
				for (int memberIndex : memberSet) {
					float delta = memberIndex == userIndex ? 1F : 0F;
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						float memberFactor = userFactors.getValue(memberIndex, factorIndex);
						float positiveFactor = itemFactors.getValue(positiveItemIndex, factorIndex);
						float negativeFactor = itemFactors.getValue(negativeItemIndex, factorIndex);
						float deltaGroup = rho * averageWeight * positiveFactor + (1 - rho) * delta * positiveFactor - delta * negativeFactor;
						userDeltas.shiftValue(memberIndex, factorIndex, learnRate * (value * deltaGroup - userRegularization * memberFactor));
						memberSums[factorIndex] += memberFactor;
					}
				}

				// update itemFactors
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactor = userFactors.getValue(userIndex, factorIndex);
					float positiveFactor = itemFactors.getValue(positiveItemIndex, factorIndex);
					float negativeFactor = itemFactors.getValue(negativeItemIndex, factorIndex);
					float positiveDelta = rho * averageWeight * memberSums[factorIndex] + (1 - rho) * userFactor;
					itemDeltas.shiftValue(positiveItemIndex, factorIndex, learnRate * (value * positiveDelta - itemRegularization * positiveFactor));
					float negativeDelta = -userFactor;
					itemDeltas.shiftValue(negativeItemIndex, factorIndex, learnRate * (value * negativeDelta - itemRegularization * negativeFactor));
				}
			}
			userFactors.addMatrix(userDeltas, false);
			itemFactors.addMatrix(itemDeltas, false);

			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			isLearned(iterationStep);
			currentLoss = totalLoss;
		}
	}

	private float predict(int userIndex, int itemIndex, Set<Integer> memberIndexes) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		DenseVector userVector = userFactors.getRowVector(userIndex);
		DenseVector itemVector = itemFactors.getRowVector(itemIndex);
		float value = itemBiases.getValue(itemIndex) + scalar.dotProduct(userVector, itemVector).getValue();
		float sum = 0F;
		for (int memberIndex : memberIndexes) {
			userVector = userFactors.getRowVector(memberIndex);
			sum += scalar.dotProduct(userVector, itemVector).getValue();
		}
		float groupRate = sum / memberIndexes.size() + itemBiases.getValue(itemIndex);
		return rho * groupRate + (1 - rho) * value;
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
