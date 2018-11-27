package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;
import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * Kabbur et al., <strong>FISM: Factored Item Similarity Models for Top-N
 * Recommender Systems</strong>, KDD 2013.
 */
// 注意:FISM使用itemFactors来组成userFactors
public class FISMrmseRecommender extends MatrixFactorizationRecommender {
	private int numNeighbors;
	private float rho, alpha, beta, gamma;

	/**
	 * bias regularization
	 */
	private float learnRatio;

	/**
	 * items and users biases vector
	 */
	private DenseVector itemBiases, userBiases;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		// 注意:FISM使用itemFactors来组成userFactors
		userFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.distributionOf(distribution));
		itemFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.distributionOf(distribution));
		userBiases = DenseVector.valueOf(numberOfUsers, VectorMapper.distributionOf(distribution));
		itemBiases = DenseVector.valueOf(numberOfItems, VectorMapper.distributionOf(distribution));
		numNeighbors = trainMatrix.getElementSize();
		rho = configuration.getFloat("rec.fismrmse.rho");// 3-15
		alpha = configuration.getFloat("rec.fismrmse.alpha", 0.5F);
		beta = configuration.getFloat("rec.fismrmse.beta", 0.6F);
		gamma = configuration.getFloat("rec.fismrmse.gamma", 0.1F);
		learnRatio = configuration.getFloat("rec.fismrmse.lrate", 0.0001F);
	}

	@Override
	protected void doPractice() {
		DefaultScalar scalar = DefaultScalar.getInstance();
		int sampleSize = (int) (rho * numNeighbors);
		int totalSize = numberOfUsers * numberOfItems;
		Table<Integer, Integer, Float> rateMatrix = HashBasedTable.create();
		for (MatrixScalar cell : trainMatrix) {
			rateMatrix.put(cell.getRow(), cell.getColumn(), cell.getValue());
		}
		int[] sampleIndexes = new int[sampleSize];

		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			DenseVector userVector = DenseVector.valueOf(numberOfFactors);
			totalLoss = 0F;
			// new training data by sampling negative values
			// R是一个在trainMatrix基础上增加负样本的矩阵.

			// make a random sample of negative feedback (total - nnz)
			for (int sampleIndex = 0; sampleIndex < sampleSize; sampleIndex++) {
				while (true) {
					int randomIndex = RandomUtility.randomInteger(totalSize - numNeighbors);
					int rowIndex = randomIndex / numberOfItems;
					int columnIndex = randomIndex % numberOfItems;
					if (rateMatrix.contains(rowIndex, columnIndex)) {
						continue;
					}
					sampleIndexes[sampleIndex] = randomIndex;
					rateMatrix.put(rowIndex, columnIndex, 0F);
					break;
				}
			}

			// update throughout each user-item-rating (u, i, rui) cell
			for (Cell<Integer, Integer, Float> cell : rateMatrix.cellSet()) {
				int userIndex = cell.getRowKey();
				int itemIndex = cell.getColumnKey();
				float rate = cell.getValue();
				SparseVector rateVector = trainMatrix.getRowVector(userIndex);
				int size = rateVector.getElementSize() - 1;
				if (size == 0 || size == -1) {
					size = 1;
				}
				for (VectorScalar term : rateVector) {
					int compareIndex = term.getIndex();
					if (itemIndex != compareIndex) {
						userVector.addVector(userFactors.getRowVector(compareIndex));
					}
				}
				userVector.scaleValues((float) Math.pow(size, -alpha));
				// for efficiency, use the below code to predict rui instead of
				// simply using "predict(u,j)"
				float itemBias = itemBiases.getValue(itemIndex);
				float predict = itemBias + scalar.dotProduct(itemFactors.getRowVector(itemIndex), userVector).getValue();
				float error = rate - predict;
				totalLoss += error * error;
				// update bi
				itemBiases.shiftValue(itemIndex, learnRatio * (error - gamma * itemBias));
				totalLoss += gamma * itemBias * itemBias;

				DenseVector factorVector = itemFactors.getRowVector(itemIndex);
				factorVector.mapValues((vectorIndex, vectorValue, message) -> {
					return (userVector.getValue(vectorIndex) * error - vectorValue * beta) * learnRatio + vectorValue;
				}, null, MathCalculator.SERIAL);
				totalLoss += beta * scalar.dotProduct(factorVector, factorVector).getValue();

				for (VectorScalar term : rateVector) {
					int compareIndex = term.getIndex();
					if (itemIndex != compareIndex) {
						float scale = (float) (error * Math.pow(size, -alpha));
						factorVector = userFactors.getRowVector(compareIndex);
						factorVector.mapValues((vectorIndex, vectorValue, message) -> {
							return (vectorValue * scale - vectorValue * beta) * learnRatio + vectorValue;
						}, null, MathCalculator.SERIAL);
						totalLoss += beta * scalar.dotProduct(factorVector, factorVector).getValue();
					}
				}
			}

			for (int sampleIndex : sampleIndexes) {
				int rowIndex = sampleIndex / numberOfItems;
				int columnIndex = sampleIndex % numberOfItems;
				rateMatrix.remove(rowIndex, columnIndex);
			}

			totalLoss *= 0.5F;
			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			currentLoss = totalLoss;
		}

	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float bias = userBiases.getValue(userIndex) + itemBiases.getValue(itemIndex);
		float sum = 0F;
		int count = 0;
		for (VectorScalar term : trainMatrix.getRowVector(userIndex)) {
			int index = term.getIndex();
			// for test, i and j will be always unequal as j is unrated
			if (index != itemIndex) {
				DenseVector userVector = userFactors.getRowVector(index);
				DenseVector itemVector = itemFactors.getRowVector(itemIndex);
				sum += scalar.dotProduct(userVector, itemVector).getValue();
				count++;
			}
		}
		sum *= (float) (count > 0 ? Math.pow(count, -alpha) : 0F);
		return bias + sum;
	}

}
