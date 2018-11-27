package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.HashMap;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * Shi et al., <strong>List-wise learning to rank with matrix factorization for
 * collaborative filtering</strong>, RecSys 2010.
 *
 * Alpha version
 * 
 * @author Yuanyuan Jin and Keqiang Wang
 */
public class ListRankMFRecommender extends MatrixFactorizationRecommender {

	private static final float nearZero = 1E-10F;

	public DenseVector userExp;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		userFactors = DenseMatrix.valueOf(numberOfUsers, numberOfFactors, MatrixMapper.randomOf(0.1F));
		itemFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.randomOf(0.1F));

		userExp = DenseVector.valueOf(numberOfUsers);
		for (MatrixScalar matrixentry : trainMatrix) {
			int userIdx = matrixentry.getRow();
			float realRating = matrixentry.getValue() / maximumOfScore;
			userExp.shiftValue(userIdx, (float) Math.exp(realRating));
		}
	}

	@Override
	protected void doPractice() {
		DefaultScalar scalar = DefaultScalar.getInstance();
		float lastLoss = getLoss(scalar);

		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			// TODO 考虑重构
			DenseMatrix userDeltas = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
			DenseMatrix itemDeltas = DenseMatrix.valueOf(numberOfItems, numberOfFactors);

			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {

				DenseVector userVector = userFactors.getRowVector(userIndex);

				float uexp = 0F;
				HashMap<Integer, Float> predictRates = new HashMap<>();
				SparseVector rateVector = trainMatrix.getRowVector(userIndex);
				for (VectorScalar term : rateVector) {
					int itemIndex = term.getIndex();
					DenseVector itemVector = itemFactors.getRowVector(itemIndex);
					float predict = scalar.dotProduct(userVector, itemVector).getValue();
					uexp += (float) Math.exp(MathUtility.logistic(predict));
					predictRates.put(itemIndex, predict);
				}

				for (VectorScalar term : rateVector) {
					int itemIndex = term.getIndex();
					float rate = term.getValue() / maximumOfScore;
					float predict = predictRates.get(itemIndex);
					float error = (float) (Math.exp(MathUtility.logistic(predict)) / uexp - Math.exp(rate) / userExp.getValue(userIndex)) * MathUtility.logisticGradientValue(predict);

					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						float userFactor = userFactors.getValue(userIndex, factorIndex);
						float itemFactor = itemFactors.getValue(itemIndex, factorIndex);
						float userGradient = error * itemFactor;
						userDeltas.shiftValue(userIndex, factorIndex, userGradient);
						float itemGradient = error * userFactor;
						itemDeltas.shiftValue(itemIndex, factorIndex, itemGradient);
					}
				}
			}

			do {
				userFactors.mapValues((row, column, value, message) -> {
					value = value + value * (-learnRate * userRegularization) + userDeltas.getValue(row, column) * -learnRate;
					return value;
				}, null, MathCalculator.PARALLEL);
				itemFactors.mapValues((row, column, value, message) -> {
					value = value + value * (-learnRate * itemRegularization) + itemDeltas.getValue(row, column) * -learnRate;
					return value;
				}, null, MathCalculator.PARALLEL);
				totalLoss = getLoss(scalar);
			} while (totalLoss > lastLoss);

			String info = " iter " + iterationStep + ": loss = " + totalLoss + ", delta_loss = " + (lastLoss - totalLoss);
			logger.info(info);

			lastLoss = totalLoss;
		} // end of training
	}

	private float getLoss(DefaultScalar scalar) {
		float loss = 0F;
		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			DenseVector userVector = userFactors.getRowVector(userIndex);
			float uexp = 0F;
			SparseVector rateVector = trainMatrix.getRowVector(userIndex);
			HashMap<Integer, Float> predictRates = new HashMap<>(rateVector.getElementSize());
			for (VectorScalar term : rateVector) {
				int itemIndex = term.getIndex();
				DenseVector itemVector = itemFactors.getRowVector(itemIndex);
				float predict = scalar.dotProduct(userVector, itemVector).getValue();;
				uexp += Math.exp(MathUtility.logistic(predict));
				predictRates.put(itemIndex, predict);
			}
			for (VectorScalar term : rateVector) {
				int itemIndex = term.getIndex();
				float rate = term.getValue() / maximumOfScore;
				float predict = predictRates.get(itemIndex);
				loss -= Math.exp(rate) / userExp.getValue(userIndex) * Math.log(Math.exp(MathUtility.logistic(predict)) / uexp);
			}
			for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
				float userFactor = userFactors.getValue(userIndex, factorIndex);
				loss += 0.5F * userRegularization * userFactor * userFactor;
			}
		}

		for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
			for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
				float itemFactor = itemFactors.getValue(itemIndex, factorIndex);
				loss += 0.5F * itemRegularization * itemFactor * itemFactor;
			}
		}
		return loss;
	}

}
