package com.jstarcraft.module.neuralnetwork.normalization;

import java.util.Map;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;

/**
 * 标准器
 * 
 * @author Birdy
 *
 */
public interface Normalizer {

	public enum Mode {

		GLOBAL,

		LOCAL;

	}

	void normalize(Map<String, MathMatrix> gradients);

}
