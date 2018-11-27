package com.jstarcraft.module.math.structure.tensor;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MathIterator;
import com.jstarcraft.module.math.structure.MathMessage;

/**
 * 数学张量
 * 
 * @author Birdy
 *
 */
public interface MathTensor extends MathIterator<TensorScalar> {

	/**
	 * 获取秩的大小
	 * 
	 * @return
	 */
	int getOrderSize();

	/**
	 * 获取指定维度的大小
	 * 
	 * @param dimension
	 * @return
	 */
	int getDimensionSize(int dimension);

	/**
	 * 获取指定维度与位置标量的索引
	 * 
	 * @param dimension
	 * @param position
	 *            范围:(0-termSize]
	 * @return
	 */
	int getIndex(int dimension, int position);

	/**
	 * 获取指定位置标量的值
	 * 
	 * @param position
	 *            范围:(0-termSize]
	 * @return
	 */
	float getValue(int position);

	/**
	 * 采集张量
	 * 
	 * @param collector
	 * @param message
	 * @param mode
	 * @return
	 */
	@Deprecated
	<T extends MathMessage> MathTensor calculate(TensorCollector<T> collector, T message, MathCalculator mode);

	/**
	 * 映射张量
	 * 
	 * @param mapper
	 * @param message
	 * @param mode
	 * @return
	 */
	@Deprecated
	<T extends MathMessage> MathTensor calculate(TensorMapper<T> mapper, T message, MathCalculator mode);

}
