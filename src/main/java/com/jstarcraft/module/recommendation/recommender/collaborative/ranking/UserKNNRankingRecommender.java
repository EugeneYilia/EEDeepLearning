package com.jstarcraft.module.recommendation.recommender.collaborative.ranking;

import java.util.Iterator;

import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.recommender.collaborative.UserKNNRecommender;

/**
 * UserKNNRecommender
 *
 * @author WangYuFeng and Keqiang Wang
 */
public class UserKNNRankingRecommender extends UserKNNRecommender {

	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		SparseVector itemVector = itemVectors[itemIndex];
		int[] neighbors = userNeighbors[userIndex];
		if (itemVector.getElementSize() == 0 || neighbors == null) {
			return 0F;
		}

		float sum = 0F, absolute = 0F;
		int count = 0;
		int leftIndex = 0, rightIndex = 0, leftSize = itemVector.getElementSize(), rightSize = neighbors.length;
		Iterator<VectorScalar> iterator = itemVector.iterator();
		VectorScalar term = iterator.next();
		// 判断两个有序数组中是否存在相同的数字
		while (leftIndex < leftSize && rightIndex < rightSize) {
			if (term.getIndex() == neighbors[rightIndex]) {
				count++;
				sum += similarityMatrix.getValue(userIndex, neighbors[rightIndex]);
				if (iterator.hasNext()) {
					term = iterator.next();
				}
				leftIndex++;
				rightIndex++;
			} else if (term.getIndex() > neighbors[rightIndex]) {
				rightIndex++;
			} else if (term.getIndex() < neighbors[rightIndex]) {
				if (iterator.hasNext()) {
					term = iterator.next();
				}
				leftIndex++;
			}
		}

		if (count == 0) {
			return 0F;
		}

		return sum;
	}

}
