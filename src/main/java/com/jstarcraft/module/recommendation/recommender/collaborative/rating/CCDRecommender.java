// Copyright (C) 2014 Guibing Guo
// This file is part of LibRec.
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
package com.jstarcraft.module.recommendation.recommender.collaborative.rating;

import java.util.Date;

import com.jstarcraft.core.utility.StringUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.matrix.MatrixMapper;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * The class implementing the Alternating Least Squares algorithm The origin
 * paper: Yunhong Zhou, Dennis Wilkinson, Robert Schreiber and Rong Pan.
 * Large-Scale Parallel Collaborative Filtering for the Netflix Prize.
 * Proceedings of the 4th international conference on Algorithmic Aspects in
 * Information and Management. Shanghai, China pp. 337-348, 2008.
 * http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/
 * netflix_aaim08(submitted).pdf
 * 
 * @author wubin (librecwb@gmail.com)
 *
 */
public class CCDRecommender extends MatrixFactorizationRecommender {

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		userFactors = DenseMatrix.valueOf(numberOfUsers, numberOfFactors, MatrixMapper.distributionOf(distribution));
		itemFactors = DenseMatrix.valueOf(numberOfItems, numberOfFactors, MatrixMapper.distributionOf(distribution));
	}

	@Override
	protected void doPractice() {
		for (int iterationStep = 1; iterationStep <= numberOfEpoches; iterationStep++) {
			for (int userIndex = 0; userIndex < numberOfUsers; userIndex++) {
				SparseVector userVector = trainMatrix.getRowVector(userIndex);
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactor = 0F;
					float numerator = 0F;
					float denominator = 0F;
					for (VectorScalar term : userVector) {
						int itemIndex = term.getIndex();
						numerator += (term.getValue() + userFactors.getValue(userIndex, factorIndex) * itemFactors.getValue(itemIndex, factorIndex)) * itemFactors.getValue(itemIndex, factorIndex);
						denominator += itemFactors.getValue(itemIndex, factorIndex) * itemFactors.getValue(itemIndex, factorIndex);
					}
					userFactor = numerator / (denominator + userRegularization);
					for (VectorScalar term : userVector) {
						int itemIndex = term.getIndex();
						term.setValue(term.getValue() - (userFactor - userFactors.getValue(userIndex, factorIndex)) * itemFactors.getValue(itemIndex, factorIndex));
					}
					userFactors.setValue(userIndex, factorIndex, userFactor);
				}
			}
			for (int itemIndex = 0; itemIndex < numberOfItems; itemIndex++) {
				SparseVector itemVector = trainMatrix.getColumnVector(itemIndex);
				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float itemFactor = 0F;
					float numerator = 0F;
					float denominator = 0F;
					for (VectorScalar term : itemVector) {
						int userIndex = term.getIndex();
						numerator += (term.getValue() + userFactors.getValue(userIndex, factorIndex) * itemFactors.getValue(itemIndex, factorIndex)) * userFactors.getValue(userIndex, factorIndex);
						denominator += userFactors.getValue(userIndex, factorIndex) * userFactors.getValue(userIndex, factorIndex);
					}
					itemFactor = numerator / (denominator + itemRegularization);
					for (VectorScalar term : itemVector) {
						int userIndex = term.getIndex();
						term.setValue(term.getValue() - (itemFactor - itemFactors.getValue(itemIndex, factorIndex)) * userFactors.getValue(userIndex, factorIndex));
					}
					itemFactors.setValue(itemIndex, factorIndex, itemFactor);
				}
			}
			logger.info(StringUtility.format("{} runs at iter {}/{} {}", this.getClass().getSimpleName(), iterationStep, numberOfEpoches, new Date()));
		}
	}

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		DefaultScalar scalar = DefaultScalar.getInstance();
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float score = scalar.dotProduct(userFactors.getRowVector(userIndex), itemFactors.getRowVector(itemIndex)).getValue();
		if (score == 0F) {
			score = meanOfScore;
		} else if (score > maximumOfScore) {
			score = maximumOfScore;
		} else if (score < minimumOfScore) {
			score = minimumOfScore;
		}
		return score;
	}

}
