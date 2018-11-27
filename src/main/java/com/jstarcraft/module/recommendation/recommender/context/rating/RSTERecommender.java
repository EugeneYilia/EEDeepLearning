package com.jstarcraft.module.recommendation.recommender.context.rating;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.SocialRecommender;

/**
 * Hao Ma, Irwin King and Michael R. Lyu, <strong>Learning to Recommend with
 * Social Trust Ensemble</strong>, SIGIR 2009.<br>
 * <p>
 * This method is quite time-consuming when dealing with the social influence
 * part.
 *
 * @author guoguibing and Keqiang Wang
 */
public class RSTERecommender extends SocialRecommender {
	private float userSocialRatio;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		userFactors = DenseMatrix.valueOf(numberOfUsers, numberOfFactors, MatrixMapper.RANDOM);
		itemFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.RANDOM);
		userSocialRatio = configuration.getFloat("rec.user.social.ratio", 0.8F);
	}

	@Override
	protected void doPractice() {
		DefaultScalar scalar = DefaultScalar.getInstance();
		DenseVector socialFactors = DenseVector.valueOf(numberOfFactors);
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			DenseMatrix userDeltas = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
			DenseMatrix itemDeltas = DenseMatrix.valueOf(numberOfItems, numberOfFactors);

			// ratings
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				SparseVector socialVector = socialMatrix.getRowVector(userIndex);
				float socialWeight = 0F;
				socialFactors.mapValues(VectorMapper.ZERO, null, MathCalculator.SERIAL);
				for (VectorScalar socialTerm : socialVector) {
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						socialFactors.setValue(factorIndex, socialFactors.getValue(factorIndex) + socialTerm.getValue() * userFactors.getValue(socialTerm.getIndex(), factorIndex));
					}
					socialWeight += socialTerm.getValue();
				}
				DenseVector userVector = userFactors.getRowVector(userIndex);
				for (VectorScalar rateTerm : trainMatrix.getRowVector(userIndex)) {
					int itemIndex = rateTerm.getIndex();
					float rate = rateTerm.getValue();
					rate = MathUtility.normalize(rate, minimumOfScore, maximumOfScore);
					// compute directly to speed up calculation
					DenseVector itemVector = itemFactors.getRowVector(itemIndex);
					float predict = scalar.dotProduct(userVector, itemVector).getValue();
					float sum = 0F;
					for (VectorScalar socialTerm : socialVector) {
						sum += socialTerm.getValue() * scalar.dotProduct(userFactors.getRowVector(socialTerm.getIndex()), itemVector).getValue();
					}
					predict = userSocialRatio * predict + (1F - userSocialRatio) * (socialWeight > 0F ? sum / socialWeight : 0F);
					float error = MathUtility.logistic(predict) - rate;
					totalLoss += error * error;
					error = MathUtility.logisticGradientValue(predict) * error;
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						float userFactor = userFactors.getValue(userIndex, factorIndex);
						float itemFactor = itemFactors.getValue(itemIndex, factorIndex);
						float userDelta = userSocialRatio * error * itemFactor + userRegularization * userFactor;
						float socialFactor = socialWeight > 0 ? socialFactors.getValue(factorIndex) / socialWeight : 0;
						float itemDelta = error * (userSocialRatio * userFactor + (1 - userSocialRatio) * socialFactor) + itemRegularization * itemFactor;
						userDeltas.shiftValue(userIndex, factorIndex, userDelta);
						itemDeltas.shiftValue(itemIndex, factorIndex, itemDelta);
						totalLoss += userRegularization * userFactor * userFactor + itemRegularization * itemFactor * itemFactor;
					}
				}
			}

			// social
			for (int trusterIndex = 0; trusterIndex < numberOfUsers; trusterIndex++) {
				SparseVector trusterVector = socialMatrix.getColumnVector(trusterIndex);
				for (VectorScalar term : trusterVector) {
					int trusteeIndex = term.getIndex();
					SparseVector trusteeVector = socialMatrix.getRowVector(trusteeIndex);
					DenseVector userVector = userFactors.getRowVector(trusteeIndex);
					float socialWeight = 0F;
					for (VectorScalar socialTerm : trusteeVector) {
						socialWeight += socialTerm.getValue();
					}
					for (VectorScalar rateTerm : trainMatrix.getRowVector(trusteeIndex)) {
						int itemIndex = rateTerm.getIndex();
						float rate = rateTerm.getValue();
						rate = MathUtility.normalize(rate, minimumOfScore, maximumOfScore);
						// compute prediction for user-item (p, j)
						DenseVector itemVector = itemFactors.getRowVector(itemIndex);
						float predict = scalar.dotProduct(userVector, itemVector).getValue();
						float sum = 0F;
						for (VectorScalar socialTerm : trusteeVector) {
							sum += socialTerm.getValue() * scalar.dotProduct(itemFactors.getRowVector(socialTerm.getIndex()), itemVector).getValue();
						}
						predict = userSocialRatio * predict + (1F - userSocialRatio) * (socialWeight > 0F ? sum / socialWeight : 0F);
						// double pred = predict(p, j, false);
						float error = MathUtility.logistic(predict) - rate;
						error = MathUtility.logisticGradientValue(predict) * error * term.getValue();
						for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
							userDeltas.shiftValue(trusterIndex, factorIndex, (1 - userSocialRatio) * error * itemFactors.getValue(itemIndex, factorIndex));
						}
					}
				}
			}
			userFactors.mapValues((row, column, value, message) -> {
				return value + userDeltas.getValue(row, column) * -learnRate;
			}, null, MathCalculator.PARALLEL);
			itemFactors.mapValues((row, column, value, message) -> {
				return value + itemDeltas.getValue(row, column) * -learnRate;
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
		DefaultScalar scalar = DefaultScalar.getInstance();
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		DenseVector userVector = userFactors.getRowVector(userIndex);
		DenseVector itemVector = itemFactors.getRowVector(itemIndex);
		float predict = scalar.dotProduct(userVector, itemVector).getValue();
		float sum = 0F, socialWeight = 0F;
		SparseVector socialVector = socialMatrix.getRowVector(userIndex);
		for (VectorScalar soicalTerm : socialVector) {
			float rate = soicalTerm.getValue();
			DenseVector soicalFactor = userFactors.getRowVector(soicalTerm.getIndex());
			sum += rate * scalar.dotProduct(soicalFactor, itemVector).getValue();
			socialWeight += rate;
		}
		predict = userSocialRatio * predict + (1 - userSocialRatio) * (socialWeight > 0 ? sum / socialWeight : 0);
		return denormalize(MathUtility.logistic(predict));
	}

}
