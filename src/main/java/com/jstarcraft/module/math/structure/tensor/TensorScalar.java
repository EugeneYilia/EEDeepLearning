package com.jstarcraft.module.math.structure.tensor;

import com.jstarcraft.module.math.structure.MathScalar;

public interface TensorScalar extends MathScalar {

	/**
	 * 获取标量指定维度的索引
	 * 
	 * @param dimension
	 * @return
	 */
	int getIndex(int dimension);

	/**
	 * 获取标量指定维度的索引
	 * 
	 * @param dimension
	 * @return
	 */
	int[] getIndexes(int[] indexes);

}
