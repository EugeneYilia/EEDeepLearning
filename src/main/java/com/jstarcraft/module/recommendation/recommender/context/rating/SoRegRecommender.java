package com.jstarcraft.module.recommendation.recommender.context.rating;

import com.jstarcraft.core.utility.ReflectionUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.matrix.SymmetryMatrix;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.SocialRecommender;
import com.jstarcraft.module.recommendation.utility.DriverUtility;
import com.jstarcraft.module.similarity.Similarity;

/**
 * Hao Ma, Dengyong Zhou, Chao Liu, Michael R. Lyu and Irwin King,
 * <strong>Recommender systems with social regularization</strong>, WSDM
 * 2011.<br>
 * <p>
 * In the original paper, this method is named as "SR2_pcc". For consistency, we
 * rename it as "SoReg" as used by some other papers such as: Tang et al.,
 * <strong>Exploiting Local and Global Social Context for
 * Recommendation</strong>, IJCAI 2013.
 *
 * @author guoguibing and Keqiang Wang
 */
public class SoRegRecommender extends SocialRecommender {

	private SymmetryMatrix socialCorrelations;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		userFactors = DenseMatrix.valueOf(numberOfUsers, numberOfFactors, MatrixMapper.RANDOM);
		itemFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.RANDOM);

		// TODO 修改为配置枚举
		Similarity correlation = ReflectionUtility.getInstance((Class<Similarity>) DriverUtility.getClass(configuration.getString("rec.similarity.class")));
		socialCorrelations = correlation.makeSimilarityMatrix(socialMatrix, false, configuration.getFloat("rec.similarity.shrinkage", 0F));

		for (MatrixScalar term : socialCorrelations) {
			float similarity = term.getValue();
			if (similarity == 0F) {
				continue;
			}
			similarity = (1F + similarity) / 2F;
			term.setValue(similarity);
		}
	}

	@Override
	protected void doPractice() {
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			totalLoss = 0F;
			DenseMatrix userDeltas = DenseMatrix.valueOf(numberOfUsers, numberOfFactors);
			DenseMatrix itemDeltas = DenseMatrix.valueOf(numberOfItems, numberOfFactors);

			// ratings
			for (MatrixScalar term : trainMatrix) {
				int userIndex = term.getRow();
				int itemIndex = term.getColumn();
				float error = predict(userIndex, itemIndex) - term.getValue();
				totalLoss += error * error;
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactorValue = userFactors.getValue(userIndex, factorIndex);
					float itemFactorValue = itemFactors.getValue(itemIndex, factorIndex);
					userDeltas.shiftValue(userIndex, factorIndex, error * itemFactorValue + userRegularization * userFactorValue);
					itemDeltas.shiftValue(itemIndex, factorIndex, error * userFactorValue + itemRegularization * itemFactorValue);
					totalLoss += userRegularization * userFactorValue * userFactorValue + itemRegularization * itemFactorValue * itemFactorValue;
				}
			}

			// friends
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				// out links: F+
				SparseVector trusterVector = socialMatrix.getRowVector(userIndex);
				for (VectorScalar term : trusterVector) {
					int trusterIndex = term.getIndex();
					float trusterSimilarity = socialCorrelations.getValue(userIndex, trusterIndex);
					if (!Float.isNaN(trusterSimilarity)) {
						for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
							float userFactor = userFactors.getValue(userIndex, factorIndex) - userFactors.getValue(trusterIndex, factorIndex);
							userDeltas.shiftValue(userIndex, factorIndex, socialRegularization * trusterSimilarity * userFactor);
							totalLoss += socialRegularization * trusterSimilarity * userFactor * userFactor;
						}
					}
				}

				// in links: F-
				SparseVector trusteeVector = socialMatrix.getColumnVector(userIndex);
				for (VectorScalar term : trusteeVector) {
					int trusteeIndex = term.getIndex();
					float trusteeSimilarity = socialCorrelations.getValue(userIndex, trusteeIndex);
					if (!Float.isNaN(trusteeSimilarity)) {
						for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
							float userFactor = userFactors.getValue(userIndex, factorIndex) - userFactors.getValue(trusteeIndex, factorIndex);
							userDeltas.shiftValue(userIndex, factorIndex, socialRegularization * trusteeSimilarity * userFactor);
							totalLoss += socialRegularization * trusteeSimilarity * userFactor * userFactor;
						}
					}
				}
			}

			// end of for loop
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
	protected float predict(int userIndex, int itemIndex) {
		float predictRating = super.predict(userIndex, itemIndex);

		if (predictRating > maximumOfScore) {
			predictRating = maximumOfScore;
		} else if (predictRating < minimumOfScore) {
			predictRating = minimumOfScore;
		}

		return predictRating;
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		return predict(userIndex, itemIndex);
	}

}
