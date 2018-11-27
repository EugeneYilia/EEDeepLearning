package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.algorithm.Probability;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * AoBPR: BPR with Adaptive Oversampling<br>
 * <p>
 * Rendle and Freudenthaler, <strong>Improving pairwise learning for item
 * recommendation from implicit feedback</strong>, WSDM 2014.
 *
 * @author guoguibing and Keqiang Wang
 */
public class AoBPRRecommender extends MatrixFactorizationRecommender {
	private int loopNumber;

	/**
	 * item geometric distribution parameter
	 */
	private int lambdaItem;

	// TODO 考虑修改为矩阵和向量
	private float[] factorVariances;
	private int[][] factorRanks;
	private Probability rankProbabilities;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		// set for this alg
		lambdaItem = (int) (configuration.getFloat("rec.item.distribution.parameter") * numberOfItems);
		// lamda_Item=500;
		loopNumber = (int) (numberOfItems * Math.log(numberOfItems));

		factorVariances = new float[numberOfFactors];
		factorRanks = new int[numberOfFactors][numberOfItems];
	}

	@Override
	protected void doPractice() {
		// 排序列表
		List<KeyValue<Integer, Float>> sortList = new ArrayList<>(numberOfItems);
		rankProbabilities = new Probability(numberOfItems, (index, value, message) -> {
			sortList.add(new KeyValue<>(index, 0F));
			return (float) Math.exp(-(index + 1) / lambdaItem);
		});
		List<Set<Integer>> userItemSet = getUserItemSet(trainMatrix);

		// TODO 此处需要重构
		List<Integer> userIndexes = new ArrayList<>(numberOfActions), itemIndexes = new ArrayList<>(numberOfActions);
		for (MatrixScalar term : trainMatrix) {
			int userIndex = term.getRow();
			int itemIndex = term.getColumn();
			userIndexes.add(userIndex);
			itemIndexes.add(itemIndex);
		}

		// randoms get a f by p(f|c)
		Probability factorProbabilities = new Probability(numberOfFactors, VectorMapper.ZERO);

		int sampleCount = 0;
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			for (int sampleIndex = 0, sampleTimes = numberOfUsers * 100; sampleIndex < sampleTimes; sampleIndex++) {
				// update Ranking every |I|log|I|
				if (sampleCount % loopNumber == 0) {
					updateSortListByFactor(sortList);
					sampleCount = 0;
				}
				sampleCount++;

				// randomly draw (u, i, j)
				int userIndex, positiveItemIndex, negativeItemIndex;
				while (true) {
					int random = RandomUtility.randomInteger(numberOfActions);
					userIndex = userIndexes.get(random);
					Set<Integer> itemSet = userItemSet.get(userIndex);
					if (itemSet.size() == 0 || itemSet.size() == numberOfItems) {
						continue;
					}
					positiveItemIndex = itemIndexes.get(random);
					// 计算概率
					DenseVector factorVector = userFactors.getRowVector(userIndex);
					factorProbabilities.calculate((index, value, message) -> {
						return Math.abs(factorVector.getValue(index)) * factorVariances[index];
					});
					do {
						// randoms get a r by exp(-r/lamda)
						int rankIndex = rankProbabilities.random();
						int factorIndex = factorProbabilities.random();
						// get the r-1 in f item
						if (userFactors.getValue(userIndex, factorIndex) > 0) {
							negativeItemIndex = factorRanks[factorIndex][rankIndex];
						} else {
							negativeItemIndex = factorRanks[factorIndex][numberOfItems - rankIndex - 1];
						}
					} while (itemSet.contains(negativeItemIndex));
					break;
				}

				// update parameters
				float positiveRate = predict(userIndex, positiveItemIndex);
				float negativeRate = predict(userIndex, negativeItemIndex);
				float error = positiveRate - negativeRate;
				float value = (float) -Math.log(MathUtility.logistic(error));
				totalLoss += value;
				value = MathUtility.logistic(-error);

				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactor = userFactors.getValue(userIndex, factorIndex);
					float positiveFactor = itemFactors.getValue(positiveItemIndex, factorIndex);
					float negativeFactor = itemFactors.getValue(negativeItemIndex, factorIndex);
					userFactors.shiftValue(userIndex, factorIndex, learnRate * (value * (positiveFactor - negativeFactor) - userRegularization * userFactor));
					itemFactors.shiftValue(positiveItemIndex, factorIndex, learnRate * (value * userFactor - itemRegularization * positiveFactor));
					itemFactors.shiftValue(negativeItemIndex, factorIndex, learnRate * (value * (-userFactor) - itemRegularization * negativeFactor));
					totalLoss += userRegularization * userFactor * userFactor + itemRegularization * positiveFactor * positiveFactor + itemRegularization * negativeFactor * negativeFactor;
				}
			}

			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			isLearned(iterationStep);
			currentLoss = totalLoss;
		}
	}

	// TODO 考虑重构
	private void updateSortListByFactor(List<KeyValue<Integer, Float>> sortList) {
		// echo for each factors
		for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
			float sum = 0F;
			DenseVector factorVector = itemFactors.getColumnVector(factorIndex);
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				float value = factorVector.getValue(itemIndex);
				sortList.get(itemIndex).setValue(value);
				sum += value;
			}
			Collections.sort(sortList, (left, right) -> {
				// 降序
				return right.getValue().compareTo(left.getValue());
			});
			float mean = sum / factorVector.getElementSize();
			sum = 0F;
			for (int sortIndex = 0; sortIndex < numberOfItems; sortIndex++) {
				float value = factorVector.getValue(sortIndex);
				sum += (value - mean) * (value - mean);
				factorRanks[factorIndex][sortIndex] = sortList.get(sortIndex).getKey();
			}
			factorVariances[factorIndex] = sum / factorVector.getElementSize();
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
