package com.jstarcraft.module.recommendation.recommender.extend;

import java.util.Iterator;

import com.jstarcraft.module.data.DataSpace;
import com.jstarcraft.module.data.accessor.InstanceAccessor;
import com.jstarcraft.module.data.accessor.SampleAccessor;
import com.jstarcraft.module.math.structure.matrix.DenseMatrix;
import com.jstarcraft.module.math.structure.tensor.SparseTensor;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.exception.RecommendationException;
import com.jstarcraft.module.recommendation.recommender.AbstractRecommender;

/**
 * Choonho Kim and Juntae Kim, <strong>A Recommendation Algorithm Using
 * Multi-Level Association Rules</strong>, WI 2003.
 * <p>
 * Simple Association Rule Recommender: we do not consider the item categories
 * (or multi levels) used in the original paper. Besides, we consider all
 * association rules without ruling out weak ones (by setting high support and
 * confidence threshold).
 *
 * @author guoguibing and wangkeqiang
 */
public class AssociationRuleRecommender extends AbstractRecommender {

	/**
	 * confidence matrix of association rules
	 */
	private DenseMatrix associationMatrix;

	/**
	 * setup
	 *
	 * @throws RecommendationException
	 *             if error occurs
	 */
	@Override
	public void prepare(Configuration configuration, SampleAccessor marker, InstanceAccessor model, DataSpace space) {
		super.prepare(configuration, marker, model, space);
		associationMatrix = DenseMatrix.valueOf(numberOfItems, numberOfItems);
	}

	@Override
	protected void doPractice() {
		// simple rule: X => Y, given that each user vector is regarded as a
		// transaction
		for (int leftItemIndex = 0; leftItemIndex < numberOfItems; leftItemIndex++) {
			// all transactions for item itemIdx
			SparseVector leftVector = trainMatrix.getColumnVector(leftItemIndex);
			int size = leftVector.getElementSize();
			for (int rightItemIndex = 0; rightItemIndex < numberOfItems; rightItemIndex++) {
				SparseVector rightVector = trainMatrix.getColumnVector(rightItemIndex);
				int leftIndex = 0, rightIndex = 0, leftSize = size, rightSize = rightVector.getElementSize();
				if (leftSize != 0 && rightSize != 0) {
					// compute confidence where containing item assoItemIdx
					// among
					// userRatingsVector
					int count = 0;
					Iterator<VectorScalar> leftIterator = leftVector.iterator();
					Iterator<VectorScalar> rightIterator = rightVector.iterator();
					VectorScalar leftTerm = leftIterator.next();
					VectorScalar rightTerm = rightIterator.next();
					// 判断两个有序数组中是否存在相同的数字
					while (leftIndex < leftSize && rightIndex < rightSize) {
						if (leftTerm.getIndex() == rightTerm.getIndex()) {
							count++;
							if (leftIterator.hasNext()) {
								leftTerm = leftIterator.next();
							}
							if (rightIterator.hasNext()) {
								rightTerm = rightIterator.next();
							}
							leftIndex++;
							rightIndex++;
						} else if (leftTerm.getIndex() > rightTerm.getIndex()) {
							if (rightIterator.hasNext()) {
								rightTerm = rightIterator.next();
							}
							rightIndex++;
						} else if (leftTerm.getIndex() < rightTerm.getIndex()) {
							if (leftIterator.hasNext()) {
								leftTerm = leftIterator.next();
							}
							leftIndex++;
						}
					}
					float value = (count + 0F) / size;
					associationMatrix.setValue(leftItemIndex, rightItemIndex, value);
				}
			}
		}
	}

	/**
	 * predict a specific rating for user userIdx on item itemIdx.
	 *
	 * @param userIndex
	 *            user index
	 * @param itemIndex
	 *            item index
	 * @return predictive rating for user userIdx on item itemIdx
	 * @throws RecommendationException
	 *             if error occurs
	 */
	@Override
	public float predict(int[] dicreteFeatures, float[] continuousFeatures) {
		int userIndex = dicreteFeatures[userDimension];
		int itemIndex = dicreteFeatures[itemDimension];
		float value = 0F;
		for (VectorScalar term : trainMatrix.getRowVector(userIndex)) {
			int associationIndex = term.getIndex();
			float association = associationMatrix.getValue(associationIndex, itemIndex);
			double rate = term.getValue();
			value += rate * association;
		}
		return value;
	}

}
