package com.jstarcraft.module.similarity;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.math.structure.matrix.SymmetryMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorScalar;

/**
 * Calculate Recommender Similarity, such as cosine, Pearson, Jaccard
 * similarity, etc.
 *
 * @author zhanghaidong
 */

public abstract class AbstractSimilarity implements Similarity {

	/**
	 * Similarity Matrix
	 */
	protected SymmetryMatrix similarityMatrix;

	protected final List<KeyValue<Float, Float>> getScoreList(MathVector leftVector, MathVector rightVector) {
		LinkedList<KeyValue<Float, Float>> scoreList = new LinkedList<>();
		int leftIndex = 0, rightIndex = 0, leftSize = leftVector.getElementSize(), rightSize = rightVector.getElementSize();
		if (leftSize != 0 && rightSize != 0) {
			Iterator<VectorScalar> leftIterator = leftVector.iterator();
			Iterator<VectorScalar> rightIterator = rightVector.iterator();
			VectorScalar leftTerm = leftIterator.next();
			VectorScalar rightTerm = rightIterator.next();
			// 判断两个有序数组中是否存在相同的数字
			while (leftIndex < leftSize && rightIndex < rightSize) {
				if (leftTerm.getIndex() == rightTerm.getIndex()) {
					scoreList.add(new KeyValue<>(leftTerm.getValue(), rightTerm.getValue()));
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
		return scoreList;
	}

	/**
	 * Build social similarity matrix with trainMatrix in dataModel.
	 *
	 * @param dataModel
	 *            the input data model
	 */
	@Override
	public SymmetryMatrix makeSimilarityMatrix(SparseMatrix scoreMatrix, boolean transpose, float scale) {
		int count = transpose ? scoreMatrix.getColumnSize() : scoreMatrix.getRowSize();
		SymmetryMatrix similarityMatrix = new SymmetryMatrix(count);
		for (int leftIndex = 0; leftIndex < count; leftIndex++) {
			SparseVector thisVector = transpose ? scoreMatrix.getColumnVector(leftIndex) : scoreMatrix.getRowVector(leftIndex);
			if (thisVector.getElementSize() == 0) {
				continue;
			}
			similarityMatrix.setValue(leftIndex, leftIndex, getIdentical());
			// user/item itself exclusive
			for (int rightIndex = leftIndex + 1; rightIndex < count; rightIndex++) {
				SparseVector thatVector = transpose ? scoreMatrix.getColumnVector(rightIndex) : scoreMatrix.getRowVector(rightIndex);
				if (thatVector.getElementSize() == 0) {
					continue;
				}
				float similarity = getCorrelation(thisVector, thatVector, scale);
				if (!Double.isNaN(similarity)) {
					similarityMatrix.setValue(leftIndex, rightIndex, similarity);
				}
			}
		}
		return similarityMatrix;
	}

	/**
	 * Build social similarity matrix with trainMatrix and socialMatrix in
	 * dataModel.
	 * 
	 * @param dataModel
	 *            the input data model
	 */
	// TODO 实现社交相似度的代码有Bug.按照现在的代码,本质是一个用户相似度矩阵的子集.(感觉应该使用用户的社交列表来计算相似度.)
	// private void buildSocialSimilarityMatrix(DataModel dataModel) {
	// SparseMatrix trainMatrix = dataModel.getDataSplitter().getTrainData();
	// SparseMatrix socialMatrix = (SparseMatrix) dataModel.getSocialDataSet();
	// int numUsers = trainMatrix.getRowSize();
	// similarityMatrix = new SymmetryMatrix(numUsers);
	// for (int userIdx = 0; userIdx < numUsers; userIdx++) {
	// SparseVector userVector = socialMatrix.row(userIdx);
	// if (userVector.size() == 0) {
	// continue;
	// }
	// SparseVector socialList = socialMatrix.column(userIdx);
	// for (VectorTerm term : socialList) {
	// int socialIdx = term.index();
	// SparseVector socialVector = socialMatrix.row(socialIdx);
	// if (socialVector.size() == 0) {
	// continue;
	// }
	//
	// double sim = getCorrelation(userVector, socialVector);
	// if (!Double.isNaN(sim)) {
	// similarityMatrix.setValue(userIdx, socialIdx, sim);
	// }
	// }
	// }
	// }

}
