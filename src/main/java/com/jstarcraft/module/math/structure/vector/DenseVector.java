package com.jstarcraft.module.math.structure.vector;

import java.util.Iterator;
import java.util.concurrent.Semaphore;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.structure.MathAccessor;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MathIterator;
import com.jstarcraft.module.math.structure.MathMessage;
import com.jstarcraft.module.math.structure.message.MessageStorage;

/**
 * 密集向量(TODO 包含非0)
 * 
 * @author Birdy
 *
 */
public class DenseVector implements MathVector {

	/** 游标 */
	private int cursor;
	/** 偏移量 */
	private int delta;
	/** 大小 */
	private int size;
	/** 数据 */
	private float[] values;

	public DenseVector(float[] data, int cursor, int delta, int size) {
		this.values = data;
		this.cursor = cursor;
		this.delta = delta;
		this.size = size;
	}

	@Override
	public int getElementSize() {
		return size;
	}

	@Override
	public int getKnownSize() {
		return getElementSize();
	}

	@Override
	public int getUnknownSize() {
		return 0;
	}

	@Override
	public MathIterator<VectorScalar> iterateElement(MathCalculator mode, MathAccessor<VectorScalar>... accessors) {
		switch (mode) {
		case SERIAL: {
			DenseVectorScalar scalar = new DenseVectorScalar();
			for (int index = 0; index < size; index++) {
				int position = cursor + index * delta;
				scalar.update(position, index);
				for (MathAccessor<VectorScalar> accessor : accessors) {
					accessor.accessScalar(scalar);
				}
			}
			return this;
		}
		default: {
			EnvironmentContext context = EnvironmentContext.getContext();
			MessageStorage storage = MathCalculator.getStorage();
			Semaphore semaphore = storage.getSemaphore();
			for (int index = 0; index < size; index++) {
				int elementIndex = index;
				int position = cursor + index * delta;
				context.doStructureByAny(position, () -> {
					DenseVectorScalar scalar = new DenseVectorScalar();
					scalar.update(position, elementIndex);
					for (MathAccessor<VectorScalar> accessor : accessors) {
						accessor.accessScalar(scalar);
					}
					semaphore.release();
				});
			}
			try {
				semaphore.acquire(size);
			} catch (Exception exception) {
				throw new RuntimeException(exception);
			}
			return this;
		}
		}
	}

	@Override
	public DenseVector setValues(float value) {
		for (int index = 0; index < size; index++) {
			int position = cursor + index * delta;
			values[position] = value;
		}
		return this;
	}

	@Override
	public DenseVector scaleValues(float value) {
		for (int index = 0; index < size; index++) {
			int position = cursor + index * delta;
			values[position] *= value;
		}
		return this;
	}

	@Override
	public DenseVector shiftValues(float value) {
		for (int index = 0; index < size; index++) {
			int position = cursor + index * delta;
			values[position] += value;
		}
		return this;
	}

	@Override
	public float getSum(boolean absolute) {
		float sum = 0F;
		if (absolute) {
			for (int index = 0; index < size; index++) {
				sum += FastMath.abs(values[cursor + index * delta]);
			}
		} else {
			for (int index = 0; index < size; index++) {
				sum += values[cursor + index * delta];
			}
		}
		return sum;
	}

	@Override
	public boolean isConstant() {
		return true;
	}

	@Override
	public int getIndex(int position) {
		return position;
	}

	@Override
	public float getValue(int position) {
		return values[cursor + position * delta];
	}

	@Override
	public void setValue(int position, float value) {
		values[cursor + position * delta] = value;
	}

	@Override
	public void scaleValue(int position, float value) {
		values[cursor + position * delta] *= value;
	}

	@Override
	public void shiftValue(int position, float value) {
		values[cursor + position * delta] += value;
	}

	@Override
	public <T extends MathMessage> DenseVector collectValues(VectorCollector<T> collector, T message, MathCalculator mode) {
		switch (mode) {
		case SERIAL: {
			for (int index = 0; index < size; index++) {
				int position = cursor + index * delta;
				collector.collect(index, values[position], message);
			}
			return this;
		}
		default: {
			EnvironmentContext context = EnvironmentContext.getContext();
			MessageStorage storage = MathCalculator.getStorage();
			Semaphore semaphore = storage.getSemaphore();
			for (int index = 0; index < size; index++) {
				int elementIndex = index;
				int position = cursor + index * delta;
				context.doStructureByAny(position, () -> {
					T copy = storage.detachMessage(message);
					collector.collect(elementIndex, values[position], copy);
					semaphore.release();
				});
			}
			try {
				semaphore.acquire(size);
				storage.attachMessage(message);
			} catch (Exception exception) {
				throw new RuntimeException(exception);
			}
			return this;
		}
		}
	}

	@Override
	public <T extends MathMessage> DenseVector mapValues(VectorMapper<T> mapper, T message, MathCalculator mode) {
		switch (mode) {
		case SERIAL: {
			for (int index = 0; index < size; index++) {
				int position = cursor + index * delta;
				values[position] = mapper.map(index, values[position], message);
			}
			return this;
		}
		default: {
			EnvironmentContext context = EnvironmentContext.getContext();
			MessageStorage storage = MathCalculator.getStorage();
			Semaphore semaphore = storage.getSemaphore();
			for (int index = 0; index < size; index++) {
				int elementIndex = index;
				int position = cursor + index * delta;
				context.doStructureByAny(position, () -> {
					T copy = storage.detachMessage(message);
					values[position] = mapper.map(elementIndex, values[position], copy);
					semaphore.release();
				});
			}
			try {
				semaphore.acquire(size);
				storage.attachMessage(message);
			} catch (Exception exception) {
				throw new RuntimeException(exception);
			}
			return this;
		}
		}
	}

	// TODO 此方法准备取消,由StructureUtility.normalize代替.
	@Deprecated
	public DenseVector normalize(VectorMapper<?> mapper) {
		float sum = 0F;
		for (int index = 0; index < size; index++) {
			int position = cursor + index * delta;
			float value = mapper.map(index, values[position], null);
			values[position] = value;
			sum += value;
		}
		for (int index = 0; index < size; index++) {
			int position = cursor + index * delta;
			values[position] /= sum;
		}
		return this;
	}

	@Override
	public boolean equals(Object object) {
		if (this == object)
			return true;
		if (object == null)
			return false;
		if (getClass() != object.getClass())
			return false;
		DenseVector that = (DenseVector) object;
		EqualsBuilder equal = new EqualsBuilder();
		equal.append(this.values, that.values);
		return equal.isEquals();
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(values);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		StringBuilder buffer = new StringBuilder();
		for (int index = 0; index < size; index++) {
			buffer.append(getValue(index)).append(", ");
		}
		buffer.append("\n");
		return buffer.toString();
	}

	@Override
	public Iterator<VectorScalar> iterator() {
		return new DenseVectorIterator();
	}

	/**
	 * Iterator over a sparse vector
	 */
	private class DenseVectorIterator implements Iterator<VectorScalar> {

		private int index;

		private final DenseVectorScalar term = new DenseVectorScalar();

		@Override
		public boolean hasNext() {
			return index < size;
		}

		@Override
		public VectorScalar next() {
			term.update(cursor + index * delta, index++);
			return term;
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}

	}

	private class DenseVectorScalar implements VectorScalar {

		private int index;

		private int cursor;

		private void update(int cursor, int index) {
			this.cursor = cursor;
			this.index = index;
		}

		@Override
		public int getIndex() {
			return index;
		}

		@Override
		public float getValue() {
			return values[cursor];
		}

		@Override
		public void scaleValue(float value) {
			values[cursor] *= value;
		}

		@Override
		public void setValue(float value) {
			values[cursor] = value;
		}

		@Override
		public void shiftValue(float value) {
			values[cursor] += value;
		}

	}

	public static DenseVector copyOf(DenseVector vector, VectorMapper<?> mapper) {
		return DenseVector.copyOf(vector, new float[vector.size], mapper);
	}

	public static DenseVector copyOf(DenseVector vector, float[] data, VectorMapper<?> mapper) {
		DenseVector instance = new DenseVector(data, 0, 1, vector.size);
		// TODO 考虑重构.
		for (int index = 0; index < vector.size; index++) {
			instance.setValue(index, mapper.map(index, vector.getValue(index), null));
		}
		return instance;
	}

	public static DenseVector valueOf(int size) {
		DenseVector instance = new DenseVector(new float[size], 0, 1, size);
		return instance;
	}

	public static DenseVector valueOf(int size, float[] data) {
		DenseVector instance = new DenseVector(data, 0, 1, size);
		return instance;
	}

	public static DenseVector valueOf(int size, VectorMapper<?> mapper) {
		return DenseVector.valueOf(size, new float[size], mapper);
	}

	public static DenseVector valueOf(int size, float[] data, VectorMapper<?> mapper) {
		DenseVector instance = new DenseVector(data, 0, 1, size);
		// TODO 考虑重构.
		for (int index = 0; index < size; index++) {
			instance.setValue(index, mapper.map(index, instance.getValue(index), null));
		}
		return instance;
	}

}
