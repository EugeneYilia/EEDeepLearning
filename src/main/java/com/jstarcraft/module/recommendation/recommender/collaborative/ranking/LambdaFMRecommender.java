package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.Iterator;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.data.processor.DataMatcher;
import com.jstarcraft.module.data.processor.DataSorter;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.tensor.SparseTensor;
import com.jstarcraft.module.math.structure.vector.ArrayVector;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.FactorizationMachineRecommender;

/**
 * 
 * YUAN et al., <strong>LambdaFM: Learning Optimal Ranking with Factorization
 * Machines Using Lambda Surrogates</strong>, CIKM 2016.
 * 
 * @author fajie yuan
 */
public abstract class LambdaFMRecommender extends FactorizationMachineRecommender {

	/** 是否自动调整学习率 */
	protected boolean isLearned;

	/** 衰减率 */
	protected float learnDecay;

	/**
	 * learn rate, maximum learning rate
	 */
	protected float learnRate, learnLimit;

	protected int lossType;

	protected int[] positiveKeys, negativeKeys;
	protected ArrayVector positiveVector, negativeVector;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		// TODO 此处代码可以消除(使用常量Marker代替或者使用binarize.threshold)
		for (MatrixScalar term : trainMatrix) {
			term.setValue(1F);
		}

		lossType = configuration.getInteger("losstype", 3);

		isLearned = configuration.getBoolean("rec.learnrate.bolddriver", false);
		learnDecay = configuration.getFloat("rec.learnrate.decay", 1.0f);
		learnRate = configuration.getFloat("rec.iterator.learnrate", 0.01f);
		learnLimit = configuration.getFloat("rec.iterator.learnrate.maximum", 1000.0f);

		biasRegularization = configuration.getFloat("rec.fm.regw0", 0.1F);
		weightRegularization = configuration.getFloat("rec.fm.regW", 0.1F);
		factorRegularization = configuration.getFloat("rec.fm.regF", 0.001F);

		positiveKeys = new int[trainTensor.getOrderSize()];
		negativeKeys = new int[trainTensor.getOrderSize()];
	}

	protected abstract float getGradientValue(DefaultScalar scalar, int[] dataPaginations, int[] dataPositions);

	@Override
	protected void doPractice() {
		DefaultScalar scalar = DefaultScalar.getInstance();
		int[] dataPaginations = new int[numberOfUsers + 1];
		int[] dataPositions = new int[trainTensor.getElementSize()];
		int size = trainTensor.getElementSize();
		for (int index = 0; index < size; index++) {
			dataPositions[index] = index;
		}
		DataMatcher dataMatcher = (paginations, positions) -> {
			for (int index = 0; index < size; index++) {
				int feature = trainTensor.getIndex(userDimension, index);
				paginations[feature + 1]++;
			}
			int cursor = size;
			for (int index = paginations.length - 1; index > 0; index--) {
				cursor -= paginations[index];
				paginations[index] = cursor;
			}
			for (int index = 0; index < size; index++) {
				int feature = trainTensor.getIndex(userDimension, index);
				positions[paginations[feature + 1]++] = index;
			}
		};
		dataMatcher.match(dataPaginations, dataPositions);

		DenseVector positiveSum = DenseVector.valueOf(numberOfFactors);
		DenseVector negativeSum = DenseVector.valueOf(numberOfFactors);

		for (int iterationStep = 0; iterationStep < numberOfEpoches; iterationStep++) {
			long totalTime = 0;
			totalLoss = 0F;
			for (int sampleIndex = 0, sampleTimes = numberOfUsers * 50; sampleIndex < sampleTimes; sampleIndex++) {
				long current = System.currentTimeMillis();
				float gradient = getGradientValue(scalar, dataPaginations, dataPositions);
				totalTime += (System.currentTimeMillis() - current);

				sum(positiveVector, positiveSum);
				sum(negativeVector, negativeSum);
				int leftIndex = 0, rightIndex = 0;
				Iterator<VectorScalar> leftIterator = positiveVector.iterator();
				Iterator<VectorScalar> rightIterator = negativeVector.iterator();
				for (int index = 0; index < trainTensor.getOrderSize(); index++) {
					VectorScalar leftTerm = leftIterator.next();
					VectorScalar rightTerm = rightIterator.next();
					leftIndex = leftTerm.getIndex();
					rightIndex = rightTerm.getIndex();
					if (leftIndex == rightIndex) {
						weightVector.shiftValue(leftIndex, learnRate * (gradient * 0F - weightRegularization * weightVector.getValue(leftIndex)));
						totalLoss += weightRegularization * weightVector.getValue(leftIndex) * weightVector.getValue(leftIndex);

						for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
							float positiveFactor = positiveSum.getValue(factorIndex) * leftTerm.getValue() - featureFactors.getValue(leftIndex, factorIndex) * leftTerm.getValue() * leftTerm.getValue();
							float negativeFactor = negativeSum.getValue(factorIndex) * rightTerm.getValue() - featureFactors.getValue(rightIndex, factorIndex) * rightTerm.getValue() * rightTerm.getValue();

							featureFactors.shiftValue(leftIndex, factorIndex, learnRate * (gradient * (positiveFactor - negativeFactor) - factorRegularization * featureFactors.getValue(leftIndex, factorIndex)));
							totalLoss += factorRegularization * featureFactors.getValue(leftIndex, factorIndex) * featureFactors.getValue(leftIndex, factorIndex);
						}
					} else {
						weightVector.shiftValue(leftIndex, learnRate * (gradient * leftTerm.getValue() - weightRegularization * weightVector.getValue(leftIndex)));
						totalLoss += weightRegularization * weightVector.getValue(leftIndex) * weightVector.getValue(leftIndex);
						weightVector.shiftValue(rightIndex, learnRate * (gradient * -rightTerm.getValue() - weightRegularization * weightVector.getValue(rightIndex)));
						totalLoss += weightRegularization * weightVector.getValue(rightIndex) * weightVector.getValue(rightIndex);

						for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
							float positiveFactor = positiveSum.getValue(factorIndex) * leftTerm.getValue() - featureFactors.getValue(leftIndex, factorIndex) * leftTerm.getValue() * leftTerm.getValue();
							featureFactors.shiftValue(leftIndex, factorIndex, learnRate * (gradient * positiveFactor - factorRegularization * featureFactors.getValue(leftIndex, factorIndex)));
							totalLoss += factorRegularization * featureFactors.getValue(leftIndex, factorIndex) * featureFactors.getValue(leftIndex, factorIndex);

							float negativeFactor = negativeSum.getValue(factorIndex) * rightTerm.getValue() - featureFactors.getValue(rightIndex, factorIndex) * rightTerm.getValue() * rightTerm.getValue();
							featureFactors.shiftValue(rightIndex, factorIndex, learnRate * (gradient * -negativeFactor - factorRegularization * featureFactors.getValue(rightIndex, factorIndex)));
							totalLoss += factorRegularization * featureFactors.getValue(rightIndex, factorIndex) * featureFactors.getValue(rightIndex, factorIndex);
						}
					}
				}
			}
			System.out.println(totalTime);

			totalLoss *= 0.5;
			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			isLearned(iterationStep);
			currentLoss = totalLoss;
		}
	}

	protected void isLearned(int iteration) {
		if (learnRate < 0F) {
			return;
		}
		if (isLearned && iteration > 1) {
			learnRate = Math.abs(currentLoss) > Math.abs(totalLoss) ? learnRate * 1.05F : learnRate * 0.5F;
		} else if (learnDecay > 0 && learnDecay < 1) {
			learnRate *= learnDecay;
		}
		// limit to max-learn-rate after update
		if (learnLimit > 0 && learnRate > learnLimit) {
			learnRate = learnLimit;
		}
	}

	private void sum(ArrayVector vector, DenseVector sum) {
		// TODO 考虑调整为向量操作.
		for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
			float value = 0F;
			for (VectorScalar term : vector) {
				value += featureFactors.getValue(term.getIndex(), factorIndex) * term.getValue();
			}
			sum.setValue(factorIndex, value);
		}
	}

}
