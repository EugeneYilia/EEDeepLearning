package com.jstarcraft.module.math.structure.tensor;

import com.jstarcraft.core.utility.KeyValue;

/**
 * 张量转换器
 * 
 * @author Birdy
 *
 */
public interface TensorTransformer<T> {

	/**
	 * 转换
	 * 
	 * @param scalar
	 * @param instance
	 */
	void transform(KeyValue<int[], Float> keyValue, T instance);

}
