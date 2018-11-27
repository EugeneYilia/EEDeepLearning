package com.jstarcraft.module.similarity;

import java.util.Iterator;

import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;

/**
 * Jaccard Similarity
 *
 * @author zhanghaidong
 */
public class JaccardSimilarity extends AbstractSimilarity {

	/**
	 * Find the common rated items by this user and that user, or the common
	 * users have rated this item or that item. And then return the similarity.
	 *
	 * @param thisVector:
	 *            the rated items by this user, or users that have rated this
	 *            item .
	 * @param thatVector:
	 *            the rated items by that user, or users that have rated that
	 *            item.
	 * @return similarity
	 */
	@Override
	public float getCorrelation(MathVector leftVector, MathVector rightVector, float scale) {
		// compute similarity
		int intersection = 0;
		int leftIndex = 0, rightIndex = 0, leftSize = leftVector.getElementSize(), rightSize = rightVector.getElementSize();
		if (leftSize != 0 && rightSize != 0) {
			Iterator<VectorScalar> leftIterator = leftVector.iterator();
			Iterator<VectorScalar> rightIterator = rightVector.iterator();
			VectorScalar leftTerm = leftIterator.next();
			VectorScalar rightTerm = rightIterator.next();
			// 判断两个有序数组中是否存在相同的数字
			while (leftIndex < leftSize && rightIndex < rightSize) {
				if (leftTerm.getIndex() == rightTerm.getIndex()) {
					intersection++;
					leftTerm = leftIterator.next();
					rightTerm = rightIterator.next();
					leftIndex++;
					rightIndex++;
				} else if (leftTerm.getIndex() > rightTerm.getIndex()) {
					rightTerm = rightIterator.next();
					rightIndex++;
				} else if (leftTerm.getIndex() < rightTerm.getIndex()) {
					leftTerm = leftIterator.next();
					leftIndex++;
				}
			}
		}
		float union = leftSize + rightSize - intersection;
		return (intersection) / union;
	}

	@Override
	public float getIdentical() {
		return 1F;
	}

}
