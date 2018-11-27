package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;
import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.DataInstance;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MathScalar;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.vector.ArrayVector;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * 
 * Gantner et al., <strong>Bayesian Personalized Ranking for Non-Uniformly
 * Sampled Items</strong>, JMLR, 2012.
 * 
 * @author guoguibing
 * 
 */
public class VBPRRecommender extends MatrixFactorizationRecommender {

	/**
	 * items biases
	 */
	private DenseVector itemBiases;

	private float biasRegularization;

	private double featureRegularization;

	private int numberOfFeatures;
	private DenseMatrix userFeatures;
	private DenseVector itemFeatures;
	private DenseMatrix featureFactors;

	private Table<Integer, Integer, Float> featureTable;
	private DenseMatrix factorMatrix;
	private DenseVector featureVector;

	/** 采样比例 */
	private int sampleRatio;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);

		// TODO 此处代码可以消除(使用常量Marker代替或者使用binarize.threshold)
		for (MatrixScalar term : trainMatrix) {
			term.setValue(1F);
		}

		biasRegularization = configuration.getFloat("rec.bias.regularization", 0.1F);
		// TODO 此处应该修改为配置或者动态计算.
		numberOfFeatures = 4096;
		featureRegularization = 1000;
		sampleRatio = configuration.getInteger("rec.vbpr.alpha", 5);

		itemBiases = DenseVector.valueOf(numberOfItems, VectorMapper.distributionOf(distribution));

		itemFeatures = DenseVector.valueOf(numberOfFeatures, VectorMapper.distributionOf(distribution));

		userFeatures = DenseMatrix.valueOf(numberOfUsers, numberOfFactors, MatrixMapper.distributionOf(distribution));

		featureFactors = DenseMatrix.valueOf(numberOfFactors, numberOfFeatures, MatrixMapper.distributionOf(distribution));

		float minimumValue = Float.MAX_VALUE;
		float maximumValue = Float.MIN_VALUE;
		featureTable = HashBasedTable.create();
		InstanceAccessor featureModel = space.getModule("article");
		for (DataInstance instance : featureModel) {
			int itemIndex = instance.getDiscreteFeature(0);
			int featureIndex = instance.getDiscreteFeature(1);
			Float featureValue = instance.getContinuousFeature(0);
			if (featureValue < minimumValue) {
				minimumValue = featureValue;
			}
			if (featureValue > maximumValue) {
				maximumValue = featureValue;
			}
			featureTable.put(itemIndex, featureIndex, featureValue);
		}
		for (Cell<Integer, Integer, Float> cell : featureTable.cellSet()) {
			float value = (cell.getValue() - minimumValue) / (maximumValue - minimumValue);
			featureTable.put(cell.getRowKey(), cell.getColumnKey(), value);
		}
		factorMatrix = DenseMatrix.valueOf(numberOfFactors, numberOfItems, MatrixMapper.distributionOf(distribution));
	}

	@Override
	protected void doPractice() {
		DefaultScalar scalar = DefaultScalar.getInstance();
		DenseVector factorVector = DenseVector.valueOf(featureFactors.getRowSize());
		ArrayVector[] featureVectors = new ArrayVector[numberOfItems];
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			Map<Integer, Float> keyValues = new TreeMap<>(featureTable.row(itemIndex));
			int[] featureIndexes = new int[keyValues.size()];
			float[] featureValues = new float[keyValues.size()];
			int index = 0;
			for (Entry<Integer, Float> keyValue : keyValues.entrySet()) {
				featureIndexes[index] = keyValue.getKey();
				featureValues[index] = keyValue.getValue();
				index++;
			}
			featureVectors[itemIndex] = new ArrayVector(numberOfFeatures, featureIndexes, featureValues);
		}
		float[] featureValues = new float[numberOfFeatures];

		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			for (int sampleIndex = 0, numberOfSamples = numberOfUsers * sampleRatio; sampleIndex < numberOfSamples; sampleIndex++) {
				// randomly draw (u, i, j)
				int userKey, positiveItemKey, negativeItemKey;
				while (true) {
					userKey = RandomUtility.randomInteger(numberOfUsers);
					SparseVector userVector = trainMatrix.getRowVector(userKey);
					if (userVector.getElementSize() == 0) {
						continue;
					}
					positiveItemKey = userVector.randomKey();
					negativeItemKey = RandomUtility.randomInteger(numberOfItems - userVector.getElementSize());
					for (VectorScalar term : userVector) {
						if (negativeItemKey >= term.getIndex()) {
							negativeItemKey++;
						} else {
							break;
						}
					}
					break;
				}
				int userIndex = userKey, positiveItemIndex = positiveItemKey, negativeItemIndex = negativeItemKey;
				ArrayVector positiveItemVector = featureVectors[positiveItemIndex];
				ArrayVector negativeItemVector = featureVectors[negativeItemIndex];
				// update parameters
				float positiveScore = predict(userIndex, positiveItemIndex, scalar.dotProduct(itemFeatures, positiveItemVector).getValue(), factorVector.dotProduct(featureFactors, false, positiveItemVector, MathCalculator.SERIAL));
				float negativeScore = predict(userIndex, negativeItemIndex, scalar.dotProduct(itemFeatures, negativeItemVector).getValue(), factorVector.dotProduct(featureFactors, false, negativeItemVector, MathCalculator.SERIAL));
				float error = MathUtility.logistic(positiveScore - negativeScore);
				totalLoss += (float) -Math.log(error);
				// update bias
				float positiveBias = itemBiases.getValue(positiveItemIndex), negativeBias = itemBiases.getValue(negativeItemIndex);
				itemBiases.shiftValue(positiveItemIndex, learnRate * (error - biasRegularization * positiveBias));
				itemBiases.shiftValue(negativeItemIndex, learnRate * (-error - biasRegularization * negativeBias));
				totalLoss += biasRegularization * positiveBias * positiveBias + biasRegularization * negativeBias * negativeBias;
				for (VectorScalar term : positiveItemVector) {
					featureValues[term.getIndex()] = term.getValue();
				}
				for (VectorScalar term : negativeItemVector) {
					featureValues[term.getIndex()] -= term.getValue();
				}
				// update user/item vectors
				// 按照因子切割任务实现并发计算.
				// CountDownLatch factorLatch = new
				// CountDownLatch(numberOfFactors);
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactor = userFactors.getValue(userIndex, factorIndex);
					float positiveItemFactor = itemFactors.getValue(positiveItemIndex, factorIndex);
					float negativeItemFactor = itemFactors.getValue(negativeItemIndex, factorIndex);
					userFactors.shiftValue(userIndex, factorIndex, learnRate * (error * (positiveItemFactor - negativeItemFactor) - userRegularization * userFactor));
					itemFactors.shiftValue(positiveItemIndex, factorIndex, learnRate * (error * (userFactor) - itemRegularization * positiveItemFactor));
					itemFactors.shiftValue(negativeItemIndex, factorIndex, learnRate * (error * (-userFactor) - itemRegularization * negativeItemFactor));
					totalLoss += userRegularization * userFactor * userFactor + itemRegularization * positiveItemFactor * positiveItemFactor + itemRegularization * negativeItemFactor * negativeItemFactor;

					float userFeature = userFeatures.getValue(userIndex, factorIndex);
					DenseVector featureVector = featureFactors.getRowVector(factorIndex);
					userFeatures.shiftValue(userIndex, factorIndex, learnRate * (error * (scalar.dotProduct(featureVector, positiveItemVector).getValue() - scalar.dotProduct(featureVector, negativeItemVector).getValue()) - userRegularization * userFeature));
					totalLoss += userRegularization * userFeature * userFeature;
					featureVector.mapValues((featureIndex, value, message) -> {
						totalLoss += featureRegularization * value * value;
						value += learnRate * (error * userFeature * featureValues[featureIndex] - featureRegularization * value);
						return value;
					}, null, MathCalculator.SERIAL);
				}
				// 按照特征切割任务实现并发计算.
				itemFeatures.mapValues((featureIndex, value, message) -> {
					totalLoss += featureRegularization * value * value;
					value += learnRate * (featureValues[featureIndex] - featureRegularization * value);
					return value;
				}, null, MathCalculator.SERIAL);
				// try {
				// factorLatch.await();
				// } catch (Exception exception) {
				// throw new LibrecException(exception);
				// }
				for (VectorScalar term : positiveItemVector) {
					featureValues[term.getIndex()] = 0F;
				}
				for (VectorScalar term : negativeItemVector) {
					featureValues[term.getIndex()] -= 0F;
				}
			}

			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			isLearned(iterationStep);
			currentLoss = totalLoss;
		}

		factorMatrix.mapValues((row, column, value, message) -> {
			ArrayVector vector = featureVectors[column];
			value = 0;
			for (VectorScalar entry : vector) {
				value += featureFactors.getValue(row, entry.getIndex()) * entry.getValue();
			}
			return value;
		}, null, MathCalculator.PARALLEL);
		featureVector = DenseVector.valueOf(numberOfItems, (index, value, message) -> {
			return scalar.dotProduct(itemFeatures, featureVectors[index]).getValue();
		});
	}

	private float predict(int userIndex, int itemIndex, float itemFeature, MathVector factorVector) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		scalar.setValue(0F);
		scalar.shiftValue(itemBiases.getValue(itemIndex) + itemFeature);
		scalar.accumulateProduct(userFactors.getRowVector(userIndex), itemFactors.getRowVector(itemIndex));
		scalar.accumulateProduct(userFeatures.getRowVector(userIndex), factorVector);
		return scalar.getValue();
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		return predict(userIndex, itemIndex, featureVector.getValue(itemIndex), factorMatrix.getColumnVector(itemIndex));
	}

}
