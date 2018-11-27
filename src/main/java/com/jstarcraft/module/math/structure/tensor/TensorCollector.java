package com.jstarcraft.module.math.structure.tensor;

import com.jstarcraft.module.math.structure.MathMessage;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;

/**
 * 张量采集器(不会导致数据变化)
 * 
 * <pre>
 * 与{@link MathMatrix}的calculate方法相关.
 * 例如:总和/平均/最大最小
 * </pre>
 * 
 * @author Birdy
 *
 */
@Deprecated
public interface TensorCollector<T extends MathMessage> {

	void collect(int[] indexes, float value, T message);

}
