package com.jstarcraft.module.math.structure.tensor;

import com.jstarcraft.module.math.structure.MathMessage;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;

/**
 * 张量映射器(会导致数据变化)
 * 
 * <pre>
 * 与{@link MathMatrix}的calculate方法相关.
 * 例如:重置,随机.
 * </pre>
 * 
 * @author Birdy
 *
 */
@Deprecated
public interface TensorMapper<T extends MathMessage> {

	float map(int[] indexes, float value, T message);

}
