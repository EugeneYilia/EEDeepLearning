package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MatrixUtility;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * Takacs and Tikk, <strong> Alternating Least Squares for Personalized Ranking
 * </strong>, RecSys 2012.
 *
 * @author guoguibing and Keqiang Wang
 */
public class RankALSRecommender extends MatrixFactorizationRecommender {
	// whether support based weighting is used ($s_i=|U_i|$) or not ($s_i=1$)
	private boolean weight;

	private DenseVector weightVector;

	private float sumSupport;

	// TODO 考虑重构到父类
	private List<Integer> userList;

	// TODO 考虑重构到父类
	private List<Integer> itemList;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		weight = configuration.getBoolean("rec.rankals.support.weight", true);
		weightVector = DenseVector.valueOf(numberOfItems);
		sumSupport = 0;
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			float supportValue = weight ? trainMatrix.getColumnScope(itemIndex) : 1F;
			weightVector.setValue(itemIndex, supportValue);
			sumSupport += supportValue;
		}

		userList = new LinkedList<>();
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			if (trainMatrix.getRowVector(userIndex).getElementSize() > 0) {
				userList.add(userIndex);
			}
		}
		userList = new ArrayList<>(userList);

		itemList = new LinkedList<>();
		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			if (trainMatrix.getColumnVector(itemIndex).getElementSize() > 0) {
				itemList.add(itemIndex);
			}
		}
		itemList = new ArrayList<>(itemList);
	}

	@Override
	protected void doPractice() {
		// 缓存特征计算,避免消耗内存
		DenseMatrix matrixCache = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);
		DenseMatrix copyCache = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);
		DenseVector vectorCache = DenseVector.valueOf(numberOfFactors);
		DenseMatrix inverseCache = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);

		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			// P step: update user vectors
			// 特征权重矩阵和特征权重向量
			DenseMatrix factorWeightMatrix = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);
			DenseVector factorWeightVector = DenseVector.valueOf(numberOfFactors);
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				float weight = weightVector.getValue(itemIndex);
				DenseVector itemVector = itemFactors.getRowVector(itemIndex);
				factorWeightMatrix.mapValues((row, column, value, message) -> {
					return value + (itemVector.getValue(row) * itemVector.getValue(column) * weight);
				}, null, MathCalculator.PARALLEL);
				factorWeightVector.mapValues((index, value, message) -> {
					return value + itemVector.getValue(index) * weight;
				}, null, MathCalculator.SERIAL);
			}

			// 用户特征矩阵,用户权重向量,用户评分向量,用户次数向量.
			DenseMatrix userDeltas = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
			DenseVector userWeights = DenseVector.valueOf(numberOfUsers);
			DenseVector userScores = DenseVector.valueOf(numberOfUsers);
			DenseVector userTimes = DenseVector.valueOf(numberOfUsers);
			// 根据物品特征构建用户特征
			for (int userIndex : userList) {
				// for each user
				SparseVector userVector = trainMatrix.getRowVector(userIndex);

				// TODO 此处考虑重构,尽量减少数组构建
				DenseMatrix factorValues = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);
				DenseMatrix copyValues = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);
				DenseVector rateValues = DenseVector.valueOf(numberOfFactors);
				DenseVector weightValues = DenseVector.valueOf(numberOfFactors);
				float weightSum = 0F, rateSum = 0F, timeSum = userVector.getElementSize();
				for (VectorScalar term : userVector) {
					int itemIndex = term.getIndex();
					float rate = term.getValue();
					// double cui = 1;
					DenseVector itemVector = itemFactors.getRowVector(itemIndex);
					factorValues.mapValues((row, column, value, message) -> {
						return value + itemVector.getValue(row) * itemVector.getValue(column);
					}, null, MathCalculator.PARALLEL);
					// ratings of unrated items will be 0
					float weight = weightVector.getValue(itemIndex) * rate;
					float value;
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						value = itemVector.getValue(factorIndex);
						userDeltas.shiftValue(userIndex, factorIndex, value);
						rateValues.shiftValue(factorIndex, value * rate);
						weightValues.shiftValue(factorIndex, value * weight);
					}

					rateSum += rate;
					weightSum += weight;
				}

				factorValues.mapValues((row, column, value, message) -> {
					return (row == column ? userRegularization : 0F) + value * sumSupport - (userDeltas.getValue(userIndex, row) * factorWeightVector.getValue(column)) - (factorWeightVector.getValue(row) * userDeltas.getValue(userIndex, column)) + (factorWeightMatrix.getValue(row, column) * timeSum);
				}, null, MathCalculator.PARALLEL);
				float rateScale = rateSum;
				float weightScale = weightSum;
				rateValues.mapValues((index, value, message) -> {
					return (value * sumSupport - userDeltas.getValue(userIndex, index) * weightScale) - (factorWeightVector.getValue(index) * rateScale) + (weightValues.getValue(index) * timeSum);
				}, null, MathCalculator.SERIAL);
				userFactors.getRowVector(userIndex).dotProduct(MatrixUtility.inverse(factorValues, copyValues, inverseCache), false, rateValues, MathCalculator.SERIAL);

				userWeights.setValue(userIndex, weightSum);
				userScores.setValue(userIndex, rateSum);
				userTimes.setValue(userIndex, timeSum);
			}

			// Q step: update item vectors
			DenseMatrix itemFactorMatrix = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);
			DenseMatrix itemTimeMatrix = DenseMatrix.valueOf(numberOfFactors, numberOfFactors);
			DenseVector itemFactorVector = DenseVector.valueOf(numberOfFactors);
			DenseVector factorValues = DenseVector.valueOf(numberOfFactors);
			for (int userIndex : userList) {
				DenseVector userVector = userFactors.getRowVector(userIndex);
				matrixCache.dotProduct(userVector, userVector, MathCalculator.SERIAL);
				itemFactorMatrix.addMatrix(matrixCache, false);
				itemTimeMatrix.mapValues((row, column, value, message) -> {
					return value + (matrixCache.getValue(row, column) * userTimes.getValue(userIndex));
				}, null, MathCalculator.PARALLEL);
				itemFactorVector.addVector(vectorCache.dotProduct(matrixCache, false, userDeltas.getRowVector(userIndex), MathCalculator.SERIAL));
				float rateSum = userScores.getValue(userIndex);
				factorValues.mapValues((index, value, message) -> {
					return value + userVector.getValue(index) * rateSum;
				}, null, MathCalculator.SERIAL);
			}

			// 根据用户特征构建物品特征
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				// for each item
				SparseVector itemVector = trainMatrix.getColumnVector(itemIndex);

				// TODO 此处考虑重构,尽量减少数组构建
				DenseVector rateValues = DenseVector.valueOf(numberOfFactors);
				DenseVector weightValues = DenseVector.valueOf(numberOfFactors);
				DenseVector timeValues = DenseVector.valueOf(numberOfFactors);
				for (VectorScalar term : itemVector) {
					int userIndex = term.getIndex();
					float rate = term.getValue();
					float weight = userWeights.getValue(userIndex);
					float time = rate * userTimes.getValue(userIndex);
					float value;
					DenseVector userVector = userFactors.getRowVector(userIndex);
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						value = userVector.getValue(factorIndex);
						rateValues.shiftValue(factorIndex, value * rate);
						weightValues.shiftValue(factorIndex, value * weight);
						timeValues.shiftValue(factorIndex, value * time);
					}
				}

				float weight = weightVector.getValue(itemIndex);
				vectorCache.dotProduct(itemFactorMatrix, false, factorWeightVector, MathCalculator.SERIAL);
				matrixCache.mapValues((row, column, value, message) -> {
					return itemFactorMatrix.getValue(row, column) * (weight + 1);
				}, null, MathCalculator.PARALLEL);
				DenseVector itemValues = itemFactors.getRowVector(itemIndex);
				vectorCache.mapValues((index, value, message) -> {
					DefaultScalar scalar = DefaultScalar.getInstance();
					value = value + (rateValues.getValue(index) * sumSupport) - weightValues.getValue(index) + (itemFactorVector.getValue(index) * weight) - (factorValues.getValue(index) * weight) + (timeValues.getValue(index) * weight);
					value = value - scalar.dotProduct(matrixCache.getRowVector(index), itemValues).getValue();
					return value;
				}, null, MathCalculator.SERIAL);
				matrixCache.mapValues((row, column, value, message) -> {
					return (row == column ? itemRegularization : 0F) + (value / (weight + 1)) * sumSupport + itemTimeMatrix.getValue(row, column) * weight - value;
				}, null, MathCalculator.PARALLEL);
				itemValues.dotProduct(MatrixUtility.inverse(matrixCache, copyCache, inverseCache), false, vectorCache, MathCalculator.SERIAL);
			}
			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			currentLoss = totalLoss;
		}
	}

}
