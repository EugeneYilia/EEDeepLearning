package com.jstarcraft.module.math.structure.matrix;

import com.jstarcraft.module.math.structure.MathScalar;

public interface MatrixScalar extends MathScalar {

	/**
	 * 获取标量所在行的索引
	 * 
	 * @return
	 */
	int getRow();

	/**
	 * 获取标量所在列的索引
	 * 
	 * @return
	 */
	int getColumn();

}
