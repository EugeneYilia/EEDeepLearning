package com.jstarcraft.module.math.structure;

import java.util.Comparator;
import java.util.Iterator;
import java.util.PriorityQueue;

import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.core.utility.KeyValue;

/**
 * 数学迭代器
 * 
 * @author Birdy
 *
 */
public interface MathIterator<T extends MathScalar> extends Iterable<T> {

	/**
	 * 获取元素(与迭代相关)的大小
	 * 
	 * @return
	 */
	int getElementSize();

	/**
	 * 获取已知标量的数量
	 * 
	 * @return
	 */
	int getKnownSize();

	/**
	 * 获取未知标量的数量
	 * 
	 * @return
	 */
	int getUnknownSize();

	/**
	 * 遍历所有元素
	 * 
	 * @param mode
	 * @param accessors
	 * @return
	 */
	MathIterator<T> iterateElement(MathCalculator mode, MathAccessor<T>... accessors);

	/**
	 * 缩放所有标量的值
	 * 
	 * @param value
	 * @return
	 */
	MathIterator<T> scaleValues(float value);

	/**
	 * 设置所有标量的值
	 * 
	 * @param value
	 * @return
	 */
	MathIterator<T> setValues(float value);

	/**
	 * 偏移所有标量的值
	 * 
	 * @param value
	 * @return
	 */
	MathIterator<T> shiftValues(float value);

	/**
	 * 获取边界
	 * 
	 * @param absolute
	 * @return
	 */
	default KeyValue<Float, Float> getBoundary(boolean absolute) {
		float maximum = Float.NEGATIVE_INFINITY;
		float minimum = Float.POSITIVE_INFINITY;
		float value = 0F;
		if (absolute) {
			for (MathScalar term : this) {
				value = FastMath.abs(term.getValue());
				if (maximum < value) {
					maximum = value;
				}
				if (minimum > value) {
					minimum = value;
				}
			}
		} else {
			for (MathScalar term : this) {
				value = term.getValue();
				if (maximum < value) {
					maximum = value;
				}
				if (minimum > value) {
					minimum = value;
				}
			}
		}
		return new KeyValue<>(minimum, maximum);
	}

	/**
	 * 获取中位数
	 * 
	 * @return
	 */
	default float getMedian(boolean absolute) {
		int count = 0;
		PriorityQueue<Float> minimumQueue = new PriorityQueue<>(new Comparator<Float>() {

			@Override
			public int compare(Float left, Float right) {
				return left.compareTo(right);
			}

		});
		PriorityQueue<Float> maximumQueue = new PriorityQueue<>(new Comparator<Float>() {

			@Override
			public int compare(Float left, Float right) {
				return right.compareTo(left);
			}

		});
		if (absolute) {
			for (MathScalar term : this) {
				if (count % 2 == 0) {
					maximumQueue.offer(FastMath.abs(term.getValue()));
					float value = maximumQueue.poll();
					minimumQueue.offer(value);
				} else {
					minimumQueue.offer(FastMath.abs(term.getValue()));
					float value = minimumQueue.poll();
					maximumQueue.offer(value);
				}
				count++;
			}
		} else {
			for (MathScalar term : this) {
				if (count % 2 == 0) {
					maximumQueue.offer(term.getValue());
					float value = maximumQueue.poll();
					minimumQueue.offer(value);
				} else {
					minimumQueue.offer(term.getValue());
					float value = minimumQueue.poll();
					maximumQueue.offer(value);
				}
				count++;
			}
		}
		if (count % 2 == 0) {
			return new Float((minimumQueue.peek() + maximumQueue.peek())) / 2F;
		} else {
			return new Float(minimumQueue.peek());
		}
	}

	/**
	 * 获取范数
	 * 
	 * @param power
	 * @return
	 */
	default float getNorm(float power) {
		// TODO 此处对称矩阵可能会存在错误,需要Override
		// 处理power为0的情况
		if (power == 0F) {
			return getElementSize();
		} else {
			float norm = 0F;
			if (power == 1F) {
				for (MathScalar term : this) {
					norm += FastMath.abs(term.getValue());
				}
				return norm;
			}
			if (power == 2F) {
				for (MathScalar term : this) {
					norm += term.getValue() * term.getValue();
				}
				return (float) Math.sqrt(norm);
			}
			// 处理power为2的倍数次方的情况
			if ((int) power == power && power % 2F == 0F) {
				for (MathScalar term : this) {
					norm += FastMath.pow(term.getValue(), power);
				}
			} else {
				for (MathScalar term : this) {
					norm += FastMath.pow(FastMath.abs(term.getValue()), power);
				}
			}
			return (float) FastMath.pow(norm, 1F / power);
		}
	}

	/**
	 * 获取总数
	 * 
	 * @param absolute
	 * @return
	 */
	default float getSum(boolean absolute) {
		// TODO 此处对称矩阵可能会存在错误,需要Override
		float sum = 0F;
		if (absolute) {
			for (MathScalar term : this) {
				sum += FastMath.abs(term.getValue());
			}
		} else {
			for (MathScalar term : this) {
				sum += term.getValue();
			}
		}
		return sum;
	}

	/**
	 * 获取方差
	 * 
	 * @return
	 */
	default KeyValue<Float, Float> getVariance() {
		// TODO 此处对称矩阵可能会存在错误,需要Override
		float mean = Float.NaN;
		float variance = Float.NaN;
		int size = 0;
		Iterator<T> iterator = this.iterator();
		if (iterator.hasNext()) {
			MathScalar term = iterator.next();
			float value = term.getValue();
			mean = value;
			size = 1;
			variance = 0;
		}
		while (iterator.hasNext()) {
			MathScalar term = iterator.next();
			float value = term.getValue();
			float delta = (value - mean);
			size++;
			mean += delta / size;
			variance += delta * (value - mean);
		}
		return new KeyValue<>(mean, variance / size);
	}

	/**
	 * 添加数据监控器
	 * 
	 * @param monitor
	 */
	default void attachMonitor(MathMonitor<T> monitor) {
	}

	/**
	 * 删除数据监控器
	 * 
	 * @param monitor
	 */
	default void detachMonitor(MathMonitor<T> monitor) {
	}

}
