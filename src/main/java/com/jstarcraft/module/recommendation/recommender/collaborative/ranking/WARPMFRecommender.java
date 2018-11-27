// Copyright (C) 2014 Guibing Guo
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.algorithm.MathUtility;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.recommender.MatrixFactorizationRecommender;

/**
 * 
 * Rendle et al., <strong>warp.
 * 
 * @author guoguibing
 * 
 *         Note for music recommendation numUsers * 300--------others 100
 */
public class WARPMFRecommender extends MatrixFactorizationRecommender {

	private int lossType;

	private float epsilon;

	private float[] orderLosses;

	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);

		lossType = configuration.getInteger("losstype", 3);
		epsilon = configuration.getFloat("epsilon");
		orderLosses = new float[numberOfItems - 1];
		float orderLoss = 0F;
		for (int orderIndex = 1; orderIndex < numberOfItems; orderIndex++) {
			orderLoss += 1D / orderIndex;
			orderLosses[orderIndex - 1] = orderLoss;
		}
		for (int rankIndex = 1; rankIndex < numberOfItems; rankIndex++) {
			orderLosses[rankIndex - 1] /= orderLoss;
		}
	}

	@Override
	protected void doPractice() {
		int Y, N;

		for (int epochIndex = 1; epochIndex <= numberOfEpoches; epochIndex++) {
			totalLoss = 0F;
			for (int sampleIndex = 0, sampleTimes = numberOfUsers * 100; sampleIndex < sampleTimes; sampleIndex++) {
				int userIndex, positiveItemIndex, negativeItemIndex;
				float positiveScore;
				float negativeScore;
				while (true) {
					userIndex = RandomUtility.randomInteger(numberOfUsers);
					SparseVector userVector = trainMatrix.getRowVector(userIndex);
					if (userVector.getElementSize() == 0 || userVector.getElementSize() == numberOfItems) {
						continue;
					}

					N = 0;
					Y = numberOfItems - trainMatrix.getRowScope(userIndex);
					positiveItemIndex = userVector.randomKey();
					positiveScore = predict(userIndex, positiveItemIndex);
					do {
						N++;
						negativeItemIndex = RandomUtility.randomInteger(numberOfItems - userVector.getElementSize());
						for (int index = 0, size = userVector.getElementSize(); index < size; index++) {
							if (negativeItemIndex >= userVector.getIndex(index)) {
								negativeItemIndex++;
								continue;
							}
							break;
						}
						negativeScore = predict(userIndex, negativeItemIndex);
					} while ((positiveScore - negativeScore > epsilon) && N < Y - 1);
					break;
				}
				// update parameters
				float error = positiveScore - negativeScore;

				float gradient = calaculateGradientValue(lossType, error);
				int orderIndex = (int) ((Y - 1) / N);
				float orderLoss = orderLosses[orderIndex];
				gradient = gradient * orderLoss;

				totalLoss += -Math.log(MathUtility.logistic(error));

				for (int factorIndex = 0; factorIndex < numberOfFactors; factorIndex++) {
					float userFactor = userFactors.getValue(userIndex, factorIndex);
					float positiveFactor = itemFactors.getValue(positiveItemIndex, factorIndex);
					float negativeFactor = itemFactors.getValue(negativeItemIndex, factorIndex);

					userFactors.shiftValue(userIndex, factorIndex, learnRate * (gradient * (positiveFactor - negativeFactor) - userRegularization * userFactor));
					itemFactors.shiftValue(positiveItemIndex, factorIndex, learnRate * (gradient * userFactor - itemRegularization * positiveFactor));
					itemFactors.shiftValue(negativeItemIndex, factorIndex, learnRate * (gradient * (-userFactor) - itemRegularization * negativeFactor));
					totalLoss += userRegularization * userFactor * userFactor + itemRegularization * positiveFactor * positiveFactor + itemRegularization * negativeFactor * negativeFactor;
				}
			}

			if (isConverged(epochIndex) && isConverged) {
				break;
			}
			isLearned(epochIndex);
			currentLoss = totalLoss;
		}
	}

}
