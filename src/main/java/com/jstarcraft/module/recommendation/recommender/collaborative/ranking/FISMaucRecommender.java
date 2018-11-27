package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * Kabbur et al., <strong>FISM: Factored Item Similarity Models for Top-N
 * Recommender Systems</strong>, KDD 2013.
 *
 * @author SunYatong
 */
// 注意:FISM使用itemFactors来组成userFactors
public class FISMaucRecommender extends MatrixFactorizationRecommender {

	private float rho, alpha, beta, gamma;

	/**
	 * bias regularization
	 */
	private float biasRegularization;

	/**
	 * items and users biases vector
	 */
	private DenseVector itemBiases;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		// 注意:FISM使用itemFactors来组成userFactors
		userFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.distributionOf(distribution));
		itemFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.distributionOf(distribution));
		// TODO
		itemBiases = DenseVector.valueOf(numberOfItems, VectorMapper.distributionOf(distribution));
		rho = configuration.getFloat("rec.fismauc.rho");// 3-15
		alpha = configuration.getFloat("rec.fismauc.alpha", 0.5F);
		beta = configuration.getFloat("rec.fismauc.beta", 0.6F);
		gamma = configuration.getFloat("rec.fismauc.gamma", 0.1F);
		biasRegularization = configuration.getFloat("rec.iteration.learnrate", 0.0001F);
		// cacheSpec = conf.get("guava.cache.spec",
		// "maximumSize=200,expireAfterAccess=2m");
	}

	@Override
	protected void doPractice() {
		DefaultScalar scalar = DefaultScalar.getInstance();
		// x <- 0
		DenseVector userVector = DenseVector.valueOf(numberOfFactors);
		// t <- (n - 1)^(-alpha) Σ pj (j!=i)
		DenseVector itemVector = DenseVector.valueOf(numberOfFactors);

		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			// for all u in C
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				SparseVector rateVector = trainMatrix.getRowVector(userIndex);
				int size = rateVector.getElementSize();
				if (size == 0 || size == 1) {
					size = 2;
				}
				// for all i in Ru+
				for (VectorScalar positiveTerm : rateVector) {
					int positiveIndex = positiveTerm.getIndex();
					userVector.mapValues(VectorMapper.ZERO, null, MathCalculator.SERIAL);
					itemVector.mapValues(VectorMapper.ZERO, null, MathCalculator.SERIAL);
					for (VectorScalar negativeTerm : rateVector) {
						int negativeIndex = negativeTerm.getIndex();
						if (positiveIndex != negativeIndex) {
							itemVector.addVector(userFactors.getRowVector(negativeIndex));
						}
					}
					itemVector.scaleValues((float) Math.pow(size - 1, -alpha));
					// Z <- SampleZeros(rho)
					int sampleSize = (int) (rho * size);
					// make a random sample of negative feedback for Ru-
					List<Integer> negativeIndexes = new LinkedList<>();
					for (int sampleIndex = 0; sampleIndex < sampleSize; sampleIndex++) {
						int negativeItemIndex = RandomUtility.randomInteger(numberOfItems - negativeIndexes.size());
						int index = 0;
						for (int negativeIndex : negativeIndexes) {
							if (negativeItemIndex >= negativeIndex) {
								negativeItemIndex++;
								index++;
							} else {
								break;
							}
						}
						negativeIndexes.add(index, negativeItemIndex);
					}

					int leftIndex = 0, rightIndex = 0, leftSize = rateVector.getElementSize(), rightSize = sampleSize;
					if (leftSize != 0 && rightSize != 0) {
						Iterator<VectorScalar> leftIterator = rateVector.iterator();
						Iterator<Integer> rightIterator = negativeIndexes.iterator();
						VectorScalar leftTerm = leftIterator.next();
						int negativeItemIndex = rightIterator.next();
						// 判断两个有序数组中是否存在相同的数字
						while (leftIndex < leftSize && rightIndex < rightSize) {
							if (leftTerm.getIndex() == negativeItemIndex) {
								if (leftIterator.hasNext()) {
									leftTerm = leftIterator.next();
								}
								rightIterator.remove();
								if (rightIterator.hasNext()) {
									negativeItemIndex = rightIterator.next();
								}
								leftIndex++;
								rightIndex++;
							} else if (leftTerm.getIndex() > negativeItemIndex) {
								if (rightIterator.hasNext()) {
									negativeItemIndex = rightIterator.next();
								}
								rightIndex++;
							} else if (leftTerm.getIndex() < negativeItemIndex) {
								if (leftIterator.hasNext()) {
									leftTerm = leftIterator.next();
								}
								leftIndex++;
							}
						}
					}

					// for all j in Z
					for (int negativeIndex : negativeIndexes) {
						// update pui puj rui ruj
						float positiveRate = positiveTerm.getValue();
						float negativeRate = 0F;
						float positiveBias = itemBiases.getValue(positiveIndex);
						float negativeBias = itemBiases.getValue(negativeIndex);
						float positiveFactor = positiveBias + scalar.dotProduct(itemFactors.getRowVector(positiveIndex), itemVector).getValue();
						float negativeFactor = negativeBias + scalar.dotProduct(itemFactors.getRowVector(negativeIndex), itemVector).getValue();

						float error = (positiveRate - negativeRate) - (positiveFactor - negativeFactor);
						totalLoss += error * error;

						// update bi bj
						itemBiases.shiftValue(positiveIndex, biasRegularization * (error - gamma * positiveBias));
						itemBiases.shiftValue(negativeIndex, biasRegularization * (error - gamma * negativeBias));

						// update qi qj
						DenseVector positiveVector = itemFactors.getRowVector(positiveIndex);
						positiveVector.mapValues((index, value, message) -> {
							return value + (itemVector.getValue(index) * error - value * beta) * biasRegularization;
						}, null, MathCalculator.SERIAL);
						DenseVector negativeVector = itemFactors.getRowVector(negativeIndex);
						negativeVector.mapValues((index, value, message) -> {
							return value - (itemVector.getValue(index) * error - value * beta) * biasRegularization;
						}, null, MathCalculator.SERIAL);

						// update x
						userVector.mapValues((index, value, message) -> {
							return value + (positiveVector.getValue(index) - negativeVector.getValue(index)) * error;
						}, null, MathCalculator.SERIAL);
					}

					float scale = (float) (Math.pow(rho, -1) * Math.pow(size - 1, -alpha));

					// for all j in Ru+\{i}
					for (VectorScalar term : rateVector) {
						int negativeIndex = term.getIndex();
						if (negativeIndex != positiveIndex) {
							// update pj
							DenseVector negativeVector = userFactors.getRowVector(negativeIndex);
							negativeVector.mapValues((index, value, message) -> {
								return (userVector.getValue(index) * scale - value * beta) * biasRegularization + value;
							}, null, MathCalculator.SERIAL);
						}
					}
				}
			}
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				double itemBias = itemBiases.getValue(itemIndex);
				totalLoss += gamma * itemBias * itemBias;
				totalLoss += beta * scalar.dotProduct(itemFactors.getRowVector(itemIndex), itemFactors.getRowVector(itemIndex)).getValue();
				totalLoss += beta * scalar.dotProduct(userFactors.getRowVector(itemIndex), userFactors.getRowVector(itemIndex)).getValue();
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
		float bias = itemBiases.getValue(itemIndex);
		float sum = 0F;
		int count = 0;
		for (VectorScalar term : trainMatrix.getRowVector(userIndex)) {
			int compareIndex = term.getIndex();
			// for test, i and j will be always unequal as j is unrated
			if (compareIndex != itemIndex) {
				DenseVector compareVector = userFactors.getRowVector(compareIndex);
				DenseVector itemVector = itemFactors.getRowVector(itemIndex);
				sum += scalar.dotProduct(compareVector, itemVector).getValue();
				count++;
			}
		}
		sum *= (float) (count > 0 ? Math.pow(count, -alpha) : 0F);
		return bias + sum;
	}

}
