package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * Shi et al., <strong>Climf: learning to maximize reciprocal rank with
 * collaborative less-is-more filtering.</strong>, RecSys 2012.
 *
 * @author Guibing Guo, Chen Ma and Keqiang Wang
 */
public class CLIMFRecommender extends MatrixFactorizationRecommender {

	private List<Set<Integer>> userItemSet;

	@Override
	protected void doPractice() {
		userItemSet = getUserItemSet(trainMatrix);

		float[] factorValues = new float[numberOfFactors];

		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				// TODO 此处应该考虑重构,不再使用itemSet
				Set<Integer> itemSet = userItemSet.get(userIndex);

				// 缓存预测值
				DenseVector predictVector = DenseVector.valueOf(itemSet.size());
				DenseVector logisticVector = DenseVector.valueOf(itemSet.size());
				int index = 0;
				for (int itemIndex : itemSet) {
					float value = predict(userIndex, itemIndex);
					predictVector.setValue(index, value);
					logisticVector.setValue(index, MathUtility.logistic(-value));
					index++;
				}
				DenseMatrix logisticMatrix = DenseMatrix.valueOf(itemSet.size(), itemSet.size());
				DenseMatrix gradientMatrix = DenseMatrix.valueOf(itemSet.size(), itemSet.size(), (row, column, value, message) -> {
					value = predictVector.getValue(row) - predictVector.getValue(column);
					float logistic = MathUtility.logistic(value);
					logisticMatrix.setValue(row, column, logistic);
					float gradient = MathUtility.logisticGradientValue(value);
					return gradient;
				});

				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float factorValue = -userRegularization * userFactors.getValue(userIndex, factorIndex);
					int leftIndex = 0;
					for (int itemIndex : itemSet) {
						float itemFactorValue = itemFactors.getValue(itemIndex, factorIndex);
						factorValue += logisticVector.getValue(leftIndex) * itemFactorValue;
						// TODO 此处应该考虑对称性减少迭代次数
						int rightIndex = 0;
						for (int compareIndex : itemSet) {
							if (compareIndex != itemIndex) {
								float compareValue = itemFactors.getValue(compareIndex, factorIndex);
								factorValue += gradientMatrix.getValue(rightIndex, leftIndex) / (1 - logisticMatrix.getValue(rightIndex, leftIndex)) * (itemFactorValue - compareValue);
							}
							rightIndex++;
						}
						leftIndex++;
					}
					factorValues[factorIndex] = factorValue;
				}

				int leftIndex = 0;
				for (int itemIndex : itemSet) {
					float logisticValue = logisticVector.getValue(leftIndex);
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						float userFactorValue = userFactors.getValue(userIndex, factorIndex);
						float itemFactorValue = itemFactors.getValue(itemIndex, factorIndex);
						float judgeValue = 1F;
						float factorValue = judgeValue * logisticValue * userFactorValue - itemRegularization * itemFactorValue;
						// TODO 此处应该考虑对称性减少迭代次数
						int rightIndex = 0;
						for (int compareIndex : itemSet) {
							if (compareIndex != itemIndex) {
								factorValue += gradientMatrix.getValue(rightIndex, leftIndex) * (judgeValue / (judgeValue - logisticMatrix.getValue(rightIndex, leftIndex)) - judgeValue / (judgeValue - logisticMatrix.getValue(leftIndex, rightIndex))) * userFactorValue;
							}
							rightIndex++;
						}
						itemFactors.shiftValue(itemIndex, factorIndex, learnRate * factorValue);
					}
					leftIndex++;
				}

				for (int factorIdx = 0; factorIdx < numberOfFactors; factorIdx++) {
					userFactors.shiftValue(userIndex, factorIdx, learnRate * factorValues[factorIdx]);
				}

				// TODO 获取预测值
				HashMap<Integer, Float> predictMap = new HashMap<>(itemSet.size());
				for (int itemIndex : itemSet) {
					float predictValue = predict(userIndex, itemIndex);
					predictMap.put(itemIndex, predictValue);
				}
				for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
					if (itemSet.contains(itemIndex)) {
						float predictValue = predictMap.get(itemIndex);
						totalLoss += (float) Math.log(MathUtility.logistic(predictValue));
						// TODO 此处应该考虑对称性减少迭代次数
						for (int compareIndex : itemSet) {
							float compareValue = predictMap.get(compareIndex);
							totalLoss += (float) Math.log(1 - MathUtility.logistic(compareValue - predictValue));
						}
					}
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						float userFactorValue = userFactors.getValue(userIndex, factorIndex);
						float itemFactorValue = itemFactors.getValue(itemIndex, factorIndex);
						totalLoss += -0.5 * (userRegularization * userFactorValue * userFactorValue + itemRegularization * itemFactorValue * itemFactorValue);
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

}
