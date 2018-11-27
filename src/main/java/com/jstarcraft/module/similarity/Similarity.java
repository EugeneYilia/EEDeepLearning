package com.jstarcraft.module.similarity;

import com.jstarcraft.module.math.structure.matrix.SparseMatrix;
import com.jstarcraft.module.math.structure.matrix.SymmetryMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;

/**
 * 相似度
 * 
 * @author Birdy
 *
 */
public interface Similarity {

	float getCorrelation(MathVector leftVector, MathVector rightVector, float scale);

	float getIdentical();

	/**
	 * 根据分数矩阵制作相似度矩阵
	 * 
	 * @param scoreMatrix
	 * @param transpose
	 * @param scale
	 * @return
	 */
	SymmetryMatrix makeSimilarityMatrix(SparseMatrix scoreMatrix, boolean transpose, float scale);

}
