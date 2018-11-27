package com.jstarcraft.module.recommendation.recommender.context.rating;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.SocialRecommender;

/**
 * Jamali and Ester, <strong>A matrix factorization technique with trust
 * propagation for recommendation in social networks</strong>, RecSys 2010.
 *
 * @author guoguibing and Keqiang Wang
 */
public class SocialMFRecommender extends SocialRecommender {

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		userFactors = DenseMatrix.valueOf(numberOfUsers, numberOfFactors, MatrixMapper.RANDOM);
		itemFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.RANDOM);
	}

	// TODO 需要重构
	@Override
	protected void doPractice() {
		DenseVector socialFactors = DenseVector.valueOf(numberOfFactors);
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			DenseMatrix userDeltas = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
			DenseMatrix itemDeltas = DenseMatrix.valueOf(numberOfItems, numberOfFactors);

			// rated items
			for (MatrixScalar term : trainMatrix) {
				int userIndex = term.getRow();
				int itemIndex = term.getColumn();
				float rate = term.getValue();
				float predict = super.predict(userIndex, itemIndex);
				float error = MathUtility.logistic(predict) - normalize(rate);
				totalLoss += error * error;
				error = MathUtility.logisticGradientValue(predict) * error;
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactor = userFactors.getValue(userIndex, factorIndex);
					float itemFactor = itemFactors.getValue(itemIndex, factorIndex);
					userDeltas.shiftValue(userIndex, factorIndex, error * itemFactor + userRegularization * userFactor);
					itemDeltas.shiftValue(itemIndex, factorIndex, error * userFactor + itemRegularization * itemFactor);
					totalLoss += userRegularization * userFactor * userFactor + itemRegularization * itemFactor * itemFactor;
				}
			}

			// social regularization
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				SparseVector trusterVector = socialMatrix.getRowVector(userIndex);
				int numTrusters = trusterVector.getElementSize();
				if (numTrusters == 0) {
					continue;
				}
				socialFactors.mapValues(VectorMapper.ZERO, null, MathCalculator.SERIAL);
				for (VectorScalar trusterTerm : trusterVector) {
					int trusterIndex = trusterTerm.getIndex();
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						socialFactors.setValue(factorIndex, socialFactors.getValue(factorIndex) + trusterTerm.getValue() * userFactors.getValue(trusterIndex, factorIndex));
					}
				}
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float error = userFactors.getValue(userIndex, factorIndex) - socialFactors.getValue(factorIndex) / numTrusters;
					userDeltas.shiftValue(userIndex, factorIndex, socialRegularization * error);
					totalLoss += socialRegularization * error * error;
				}

				// those who trusted user u
				SparseVector trusteeVector = socialMatrix.getColumnVector(userIndex);
				int numTrustees = trusteeVector.getElementSize();
				for (VectorScalar trusteeTerm : trusteeVector) {
					int trusteeIndex = trusteeTerm.getIndex();
					trusterVector = socialMatrix.getRowVector(trusteeIndex);
					numTrusters = trusterVector.getElementSize();
					if (numTrusters == 0) {
						continue;
					}
					socialFactors.mapValues(VectorMapper.ZERO, null, MathCalculator.SERIAL);
					for (VectorScalar trusterTerm : trusterVector) {
						int trusterIndex = trusterTerm.getIndex();
						for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
							socialFactors.setValue(factorIndex, socialFactors.getValue(factorIndex) + trusterTerm.getValue() * userFactors.getValue(trusterIndex, factorIndex));
						}
					}
					for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
						userDeltas.shiftValue(userIndex, factorIndex, -socialRegularization * (trusteeTerm.getValue() / numTrustees) * (userFactors.getValue(trusteeIndex, factorIndex) - socialFactors.getValue(factorIndex) / numTrusters));
					}
				}
			}
			// update user factors
			userFactors.mapValues((row, column, value, message) -> {
				return value + userDeltas.getValue(row, column) * -learnRate;
			}, null, MathCalculator.PARALLEL);
			itemFactors.mapValues((row, column, value, message) -> {
				return value + itemDeltas.getValue(row, column) * -learnRate;
			}, null, MathCalculator.PARALLEL);

			totalLoss *= 0.5D;
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
		return denormalize(MathUtility.logistic(predict));
	}

}
