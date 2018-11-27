package com.jstarcraft.module.math.structure.vector;

import com.jstarcraft.module.math.structure.MathMessage;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.message.AccumulationMessage;

/**
 * 向量采集器(不会导致数据变化)
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
public interface VectorCollector<T extends MathMessage> {

	@Deprecated
	public static final VectorCollector<AccumulationMessage<?>> ACCUMULATOR = new VectorCollector<AccumulationMessage<?>>() {

		@Override
		public void collect(int index, float value, AccumulationMessage<?> message) {
			message.accumulateValue(value);
		}

	};

	void collect(int index, float value, T message);

}
