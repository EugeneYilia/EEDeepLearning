package com.jstarcraft.module.math.structure.vector;

import java.util.concurrent.CountDownLatch;

import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MathIterator;
import com.jstarcraft.module.math.structure.MathMessage;
import com.jstarcraft.module.math.structure.MathScalar;
import com.jstarcraft.module.math.structure.DefaultScalar;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;

/**
 * 数学向量
 * 
 * @author Birdy
 *
 */
public interface MathVector extends MathIterator<VectorScalar> {

	/**
	 * 是否位置等于索引
	 * 
	 * <pre>
	 * 与getIndex,getValue,setValue,scaleValue,shiftValue相关
	 * </pre>
	 * 
	 * @return
	 */
	boolean isConstant();

	/**
	 * 获取指定位置标量的索引
	 * 
	 * @param position
	 *            范围:(0-termSize]
	 * @return
	 */
	int getIndex(int position);

	/**
	 * 获取指定位置标量的值
	 * 
	 * @param position
	 *            范围:(0-termSize]
	 * @return
	 */
	float getValue(int position);

	/**
	 * 设置指定位置标量的值
	 * 
	 * @param position
	 * @param value
	 */
	void setValue(int position, float value);

	/**
	 * 缩放指定位置标量的值
	 * 
	 * @param position
	 * @param value
	 */
	void scaleValue(int position, float value);

	/**
	 * 偏移指定位置标量的值
	 * 
	 * @param position
	 * @param value
	 */
	void shiftValue(int position, float value);

	/**
	 * 采集向量
	 * 
	 * @param collector
	 * @param message
	 * @param mode
	 * @return
	 */
	@Deprecated
	<T extends MathMessage> MathVector collectValues(VectorCollector<T> collector, T message, MathCalculator mode);

	/**
	 * 映射向量
	 * 
	 * @param mapper
	 * @param message
	 * @param mode
	 * @return
	 */
	@Deprecated
	<T extends MathMessage> MathVector mapValues(VectorMapper<T> mapper, T message, MathCalculator mode);

	/**
	 * 向量加法运算
	 * 
	 * @param matrix
	 * @param transpose
	 * @return
	 */
	default MathVector addVector(MathVector vector) {
		for (int position = 0, size = getElementSize(); position < size; position++) {
			// TODO 此处考虑是否改为 addValue
			shiftValue(position, vector.getValue(position));
		}
		return this;
	}

	/**
	 * 向量减法运算
	 * 
	 * @param matrix
	 * @param transpose
	 * @return
	 */
	default MathVector subtractVector(MathVector vector) {
		for (int position = 0, size = getElementSize(); position < size; position++) {
			// TODO 此处考虑是否改为 subtractValue
			shiftValue(position, -vector.getValue(position));
		}
		return this;
	}

	/**
	 * 向量乘法运算
	 * 
	 * @param matrix
	 * @param transpose
	 * @return
	 */
	default MathVector multiplyVector(MathVector vector) {
		assert this.getElementSize() == vector.getElementSize();
		for (int position = 0, size = getElementSize(); position < size; position++) {
			// TODO 此处考虑是否改为 multiplyValue
			scaleValue(position, vector.getValue(position));
		}
		return this;
	}

	/**
	 * 向量除法运算
	 * 
	 * @param matrix
	 * @param transpose
	 * @return
	 */
	default MathVector divideVector(MathVector vector) {
		assert this.getElementSize() == vector.getElementSize();
		for (int position = 0, size = getElementSize(); position < size; position++) {
			// TODO 此处考虑是否改为divideValue
			scaleValue(position, 1F / vector.getValue(position));
		}
		return this;
	}

	/**
	 * 向量拷贝运算
	 * 
	 * @param matrix
	 * @param transpose
	 * @return
	 */
	default MathVector copyVector(MathVector vector) {
		assert this.getElementSize() == vector.getElementSize();
		for (int position = 0, size = getElementSize(); position < size; position++) {
			setValue(position, vector.getValue(position));
		}
		return this;
	}

	/**
	 * 向量点积运算
	 * 
	 * @param leftMatrix
	 * @param transpose
	 * @param rightVector
	 * @param mode
	 * @return
	 */
	default MathVector dotProduct(MathMatrix leftMatrix, boolean transpose, MathVector rightVector, MathCalculator mode) {
		switch (mode) {
		case SERIAL: {
			for (VectorScalar term : this) {
				int index = term.getIndex();
				MathVector leftVector = transpose ? leftMatrix.getColumnVector(index) : leftMatrix.getRowVector(index);
				term.dotProduct(leftVector, rightVector);
			}
			return this;
		}
		default: {
			int size = this.getElementSize();
			EnvironmentContext context = EnvironmentContext.getContext();
			CountDownLatch latch = new CountDownLatch(size);
			for (int position = 0; position < size; position++) {
				int index = this.getIndex(position);
				int cursor = position;
				context.doStructureByAny(position, () -> {
					MathVector leftVector = transpose ? leftMatrix.getColumnVector(index) : leftMatrix.getRowVector(index);
					DefaultScalar scalar = DefaultScalar.getInstance();
					float value = scalar.dotProduct(leftVector, rightVector).getValue();
					this.setValue(cursor, value);
					latch.countDown();
				});
			}
			try {
				latch.await();
			} catch (Exception exception) {
				throw new RuntimeException(exception);
			}
			return this;
		}
		}
	}

	/**
	 * 向量点积运算
	 * 
	 * @param leftVector
	 * @param rightMatrix
	 * @param transpose
	 * @param mode
	 * @return
	 */
	default MathVector dotProduct(MathVector leftVector, MathMatrix rightMatrix, boolean transpose, MathCalculator mode) {
		switch (mode) {
		case SERIAL: {
			for (VectorScalar term : this) {
				int index = term.getIndex();
				MathVector rightVector = transpose ? rightMatrix.getRowVector(index) : rightMatrix.getColumnVector(index);
				term.dotProduct(leftVector, rightVector);
			}
			return this;
		}
		default: {
			int size = this.getElementSize();
			EnvironmentContext context = EnvironmentContext.getContext();
			CountDownLatch latch = new CountDownLatch(size);
			for (int position = 0; position < size; position++) {
				int index = this.getIndex(position);
				int cursor = position;
				context.doStructureByAny(position, () -> {
					MathVector rightVector = transpose ? rightMatrix.getRowVector(index) : rightMatrix.getColumnVector(index);
					DefaultScalar scalar = DefaultScalar.getInstance();
					float value = scalar.dotProduct(leftVector, rightVector).getValue();
					this.setValue(cursor, value);
					latch.countDown();
				});
			}
			try {
				latch.await();
			} catch (Exception exception) {
				throw new RuntimeException(exception);
			}
			return this;
		}
		}
	}

	/**
	 * 向量累积运算
	 * 
	 * @param leftMatrix
	 * @param transpose
	 * @param rightVector
	 * @param mode
	 * @return
	 */
	default MathVector accumulateProduct(MathMatrix leftMatrix, boolean transpose, MathVector rightVector, MathCalculator mode) {
		switch (mode) {
		case SERIAL: {
			for (VectorScalar term : this) {
				int index = term.getIndex();
				MathVector leftVector = transpose ? leftMatrix.getColumnVector(index) : leftMatrix.getRowVector(index);
				term.accumulateProduct(leftVector, rightVector);
			}
			return this;
		}
		default: {
			int size = this.getElementSize();
			EnvironmentContext context = EnvironmentContext.getContext();
			CountDownLatch latch = new CountDownLatch(size);
			for (int position = 0; position < size; position++) {
				int index = this.getIndex(position);
				int cursor = position;
				context.doStructureByAny(position, () -> {
					MathVector leftVector = transpose ? leftMatrix.getColumnVector(index) : leftMatrix.getRowVector(index);
					DefaultScalar scalar = DefaultScalar.getInstance();
					float value = scalar.dotProduct(leftVector, rightVector).getValue();
					this.shiftValue(cursor, value);
					latch.countDown();
				});
			}
			try {
				latch.await();
			} catch (Exception exception) {
				throw new RuntimeException(exception);
			}
			return this;
		}
		}
	}

	/**
	 * 向量累积运算
	 * 
	 * @param leftVector
	 * @param rightMatrix
	 * @param transpose
	 * @param mode
	 * @return
	 */
	default MathVector accumulateProduct(MathVector leftVector, MathMatrix rightMatrix, boolean transpose, MathCalculator mode) {
		switch (mode) {
		case SERIAL: {
			for (VectorScalar term : this) {
				int index = term.getIndex();
				MathVector rightVector = transpose ? rightMatrix.getRowVector(index) : rightMatrix.getColumnVector(index);
				term.accumulateProduct(leftVector, rightVector);
			}
			return this;
		}
		default: {
			int size = this.getElementSize();
			EnvironmentContext context = EnvironmentContext.getContext();
			CountDownLatch latch = new CountDownLatch(size);
			for (int position = 0; position < size; position++) {
				int index = this.getIndex(position);
				int cursor = position;
				context.doStructureByAny(position, () -> {
					MathVector rightVector = transpose ? rightMatrix.getRowVector(index) : rightMatrix.getColumnVector(index);
					DefaultScalar scalar = DefaultScalar.getInstance();
					float value = scalar.dotProduct(leftVector, rightVector).getValue();
					this.shiftValue(cursor, value);
					latch.countDown();
				});
			}
			try {
				latch.await();
			} catch (Exception exception) {
				throw new RuntimeException(exception);
			}
			return this;
		}
		}
	}

}
