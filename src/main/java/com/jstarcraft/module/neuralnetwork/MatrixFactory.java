package com.jstarcraft.module.neuralnetwork;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;

/**
 * 缓存工厂
 * 
 * @author Birdy
 *
 */
public interface MatrixFactory {

	MathMatrix makeCache(int rowSize, int columnSize);

}
