package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.vector.ArrayVector;
import com.jstarcraft.module.math.structure.vector.DenseVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * Daniel D. Lee and H. Sebastian Seung, <strong>Algorithms for Non-negative
 * Matrix Factorization</strong>, NIPS 2001.
 *
 * @author guoguibing and Keqiang Wang
 */
public class NMFRecommender extends MatrixFactorizationRecommender {

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		userFactors = DenseMatrix.valueOf(numberOfUsers, numberOfFactors, MatrixMapper.randomOf(0.01F));
		itemFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.randomOf(0.01F));
	}

	@Override
	protected void doPractice() {
		DefaultScalar scalar = DefaultScalar.getInstance();
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; ++iterationStep) {
			// update userFactors by fixing itemFactors
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				SparseVector userVector = trainMatrix.getRowVector(userIndex);
				if (userVector.getElementSize() == 0) {
					continue;
				}
				int user = userIndex;
				ArrayVector predictVector = new ArrayVector(userVector, (index, value, message) -> {
					return predict(user, index);
				});
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					DenseVector factorVector = itemFactors.getColumnVector(factorIndex);
					float rate = scalar.dotProduct(factorVector, userVector).getValue();
					float predict = scalar.dotProduct(factorVector, predictVector).getValue() + MathUtility.EPSILON;
					userFactors.setValue(userIndex, factorIndex, userFactors.getValue(userIndex, factorIndex) * (rate / predict));
				}
			}

			// update itemFactors by fixing userFactors
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				SparseVector itemVector = trainMatrix.getColumnVector(itemIndex);
				if (itemVector.getElementSize() == 0) {
					continue;
				}
				int item = itemIndex;
				ArrayVector predictVector = new ArrayVector(itemVector, (index, value, message) -> {
					return predict(index, item);
				});
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					DenseVector factorVector = userFactors.getColumnVector(factorIndex);
					float rate = scalar.dotProduct(factorVector, itemVector).getValue();
					float predict = scalar.dotProduct(factorVector, predictVector).getValue() + MathUtility.EPSILON;
					itemFactors.setValue(itemIndex, factorIndex, itemFactors.getValue(itemIndex, factorIndex) * (rate / predict));
				}
			}

			// compute errors
			totalLoss = 0F;
			for (MatrixScalar term : trainMatrix) {
				int userIndex = term.getRow();
				int itemIndex = term.getColumn();
				float rate = term.getValue();
				if (rate > 0) {
					float error = predict(userIndex, itemIndex) - rate;
					totalLoss += error * error;
				}
			}
			totalLoss *= 0.5F;
			if (isConverged(iterationStep) && isConverged) {
				break;
			}
			currentLoss = totalLoss;
		}
	}

}
