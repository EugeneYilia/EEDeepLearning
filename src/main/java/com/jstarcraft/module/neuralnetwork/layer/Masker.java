package com.jstarcraft.module.neuralnetwork.layer;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;

/**
 * 掩盖器(本质为Dropout)
 * 
 * @author Birdy
 *
 */
public interface Masker {

	/**
	 * 掩盖
	 * 
	 * @param middleData
	 * @param iteration
	 * @param epoch
	 */
	void mask(MathMatrix middleData, int iteration, int epoch);

}
