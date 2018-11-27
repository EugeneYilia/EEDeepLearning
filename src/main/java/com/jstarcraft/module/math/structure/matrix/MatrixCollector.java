package com.jstarcraft.module.math.structure.matrix;

import com.jstarcraft.module.math.structure.MathMessage;
import com.jstarcraft.module.math.structure.message.AccumulationMessage;

/**
 * 矩阵采集器(不会导致数据变化)
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
public interface MatrixCollector<T extends MathMessage> {

	@Deprecated
	public static final MatrixCollector<AccumulationMessage<?>> ACCUMULATOR = new MatrixCollector<AccumulationMessage<?>>() {

		@Override
		public void collect(int row, int column, float value, AccumulationMessage<?> message) {
			message.accumulateValue(value);
		}

	};

	void collect(int row, int column, float value, T message);

}
