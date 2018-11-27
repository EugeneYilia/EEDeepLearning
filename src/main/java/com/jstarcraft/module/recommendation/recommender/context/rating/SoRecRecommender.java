package com.jstarcraft.module.recommendation.recommender.context.rating;

import java.util.ArrayList;
import java.util.List;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.SocialRecommender;

/**
 * Jamali and Ester, <strong>A matrix factorization technique with trust
 * propagation for recommendation in social networks</strong>, RecSys 2010.
 *
 * @author guoguibing and Keqiang Wang
 */
public class SoRecRecommender extends SocialRecommender {
	/**
	 * adaptive learn rate
	 */
	private DenseMatrix socialFactors;

	private float regRate, regSocial;

	private List<Integer> inDegrees, outDegrees;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		userFactors = DenseMatrix.valueOf(numberOfUsers, numberOfFactors, MatrixMapper.RANDOM);
		itemFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.RANDOM);
		socialFactors = DenseMatrix.valueOf(numberOfUsers, numberOfFactors, MatrixMapper.RANDOM);

		regRate = configuration.getFloat("rec.rate.social.regularization", 0.01F);
		regSocial = configuration.getFloat("rec.user.social.regularization", 0.01F);

		inDegrees = new ArrayList<>(numberOfUsers);
		outDegrees = new ArrayList<>(numberOfUsers);

		for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
			int in = socialMatrix.getColumnScope(userIndex);
			int out = socialMatrix.getRowScope(userIndex);
			inDegrees.add(in);
			outDegrees.add(out);
		}
	}

	@Override
	protected void doPractice() {
		DefaultScalar scalar = DefaultScalar.getInstance();
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			DenseMatrix userDeltas = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
			DenseMatrix itemDeltas = DenseMatrix.valueOf(numberOfItems, numberOfFactors);
			DenseMatrix socialDeltas = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);

			// ratings
			for (MatrixScalar term : trainMatrix) {
				int userIdx = term.getRow();
				int itemIdx = term.getColumn();
				float rate = term.getValue();
				float predict = super.predict(userIdx, itemIdx);
				float error = MathUtility.logistic(predict) - MathUtility.normalize(rate, minimumOfScore, maximumOfScore);
				totalLoss += error * error;
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactor = userFactors.getValue(userIdx, factorIndex);
					float itemFactor = itemFactors.getValue(itemIdx, factorIndex);
					userDeltas.shiftValue(userIdx, factorIndex, MathUtility.logisticGradientValue(predict) * error * itemFactor + userRegularization * userFactor);
					itemDeltas.shiftValue(itemIdx, factorIndex, MathUtility.logisticGradientValue(predict) * error * userFactor + itemRegularization * itemFactor);
					totalLoss += userRegularization * userFactor * userFactor + itemRegularization * itemFactor * itemFactor;
				}
			}

			// friends
			// TODO 此处是对称矩阵,是否有方法减少计算?
			for (MatrixScalar term : socialMatrix) {
				int userIndex = term.getRow();
				int socialIndex = term.getColumn();
				float socialRate = term.getValue();
				// tuv ~ cik in the original paper
				if (socialRate == 0F) {
					continue;
				}
				float socialPredict = scalar.dotProduct(userFactors.getRowVector(userIndex), socialFactors.getRowVector(socialIndex)).getValue();
				float socialInDegree = inDegrees.get(socialIndex); // ~ d-(k)
				float userOutDegree = outDegrees.get(userIndex); // ~ d+(i)
				float weight = (float) Math.sqrt(socialInDegree / (userOutDegree + socialInDegree));
				float socialError = MathUtility.logistic(socialPredict) - weight * socialRate;
				totalLoss += regRate * socialError * socialError;

				socialPredict = MathUtility.logisticGradientValue(socialPredict);
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactor = userFactors.getValue(userIndex, factorIndex);
					float socialFactor = socialFactors.getValue(socialIndex, factorIndex);
					userDeltas.shiftValue(userIndex, factorIndex, regRate * socialPredict * socialError * socialFactor);
					socialDeltas.shiftValue(socialIndex, factorIndex, regRate * socialPredict * socialError * userFactor + regSocial * socialFactor);
					totalLoss += regSocial * socialFactor * socialFactor;
				}
			}

			userFactors.mapValues((row, column, value, message) -> {
				return value + userDeltas.getValue(row, column) * -learnRate;
			}, null, MathCalculator.PARALLEL);
			itemFactors.mapValues((row, column, value, message) -> {
				return value + itemDeltas.getValue(row, column) * -learnRate;
			}, null, MathCalculator.PARALLEL);
			socialFactors.mapValues((row, column, value, message) -> {
				return value + socialDeltas.getValue(row, column) * -learnRate;
			}, null, MathCalculator.PARALLEL);

			totalLoss *= 0.5F;
			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			isLearned(iterationStep);
			currentLoss = totalLoss;
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float predict = super.predict(userIndex, itemIndex);
		predict = denormalize(MathUtility.logistic(predict));
		return predict;
	}

}
