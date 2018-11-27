package com.jstarcraft.module.math.structure.vector;

import java.util.Iterator;
import java.util.WeakHashMap;
import java.util.concurrent.Semaphore;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.structure.MathAccessor;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MathIterator;
import com.jstarcraft.module.math.structure.MathMessage;
import com.jstarcraft.module.math.structure.MathMonitor;
import com.jstarcraft.module.math.structure.message.MessageStorage;

/**
 * 数组向量
 * 
 * <pre>
 * 提供比稀疏向量更快的访问速度,且能够在一定范围变化索引.
 * </pre>
 * 
 * @author Birdy
 *
 */
public class ArrayVector implements MathVector {

	private int capacity;

	private int size;

	/** 索引 */
	private int[] indexes;

	/** 值 */
	private float[] values;

	private transient WeakHashMap<MathMonitor<VectorScalar>, Object> monitors = new WeakHashMap<>();

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
		return capacity - getElementSize();
	}
	
	@Override
	public MathIterator<VectorScalar> iterateElement(MathCalculator mode, MathAccessor<VectorScalar>... accessors) {
		switch (mode) {
		case SERIAL: {
			ArrayVectorScalar scalar = new ArrayVectorScalar();
			for (int position = 0; position < size; position++) {
				scalar.update(position);
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
			for (int position = 0; position < size; position++) {
				int index = position;
				context.doStructureByAny(position, () -> {
					ArrayVectorScalar scalar = new ArrayVectorScalar();
					scalar.update(index);
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
	public ArrayVector setValues(float value) {
		for (int position = 0; position < size; position++) {
			values[position] = value;
		}
		return this;
	}

	@Override
	public ArrayVector scaleValues(float value) {
		for (int position = 0; position < size; position++) {
			values[position] *= value;
		}
		return this;
	}

	@Override
	public ArrayVector shiftValues(float value) {
		for (int position = 0; position < size; position++) {
			values[position] += value;
		}
		return this;
	}

	@Override
	public float getSum(boolean absolute) {
		float sum = 0F;
		if (absolute) {
			for (int position = 0; position < size; position++) {
				sum += FastMath.abs(values[position]);
			}
		} else {
			for (int position = 0; position < size; position++) {
				sum += values[position];
			}
		}
		return sum;
	}

	@Override
	public void attachMonitor(MathMonitor<VectorScalar> monitor) {
		monitors.put(monitor, null);
	}

	@Override
	public void detachMonitor(MathMonitor<VectorScalar> monitor) {
		monitors.remove(monitor);
	}

	@Override
	public boolean isConstant() {
		return false;
	}

	@Override
	public int getIndex(int position) {
		return indexes[position];
	}

	@Override
	public float getValue(int position) {
		return values[position];
	}

	@Override
	public void setValue(int position, float value) {
		values[position] = value;
	}

	@Override
	public void scaleValue(int position, float value) {
		values[position] *= value;
	}

	@Override
	public void shiftValue(int position, float value) {
		values[position] += value;
	}

	@Override
	public <T extends MathMessage> ArrayVector collectValues(VectorCollector<T> collector, T message, MathCalculator mode) {
		switch (mode) {
		case SERIAL: {
			for (int position = 0; position < size; position++) {
				collector.collect(indexes[position], values[position], message);
			}
			return this;
		}
		default: {
			EnvironmentContext context = EnvironmentContext.getContext();
			MessageStorage storage = MathCalculator.getStorage();
			Semaphore semaphore = storage.getSemaphore();
			for (int position = 0; position < size; position++) {
				int index = position;
				context.doStructureByAny(position, () -> {
					T copy = storage.detachMessage(message);
					collector.collect(indexes[index], values[index], copy);
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
	public <T extends MathMessage> ArrayVector mapValues(VectorMapper<T> mapper, T message, MathCalculator mode) {
		switch (mode) {
		case SERIAL: {
			for (int position = 0; position < size; position++) {
				values[position] = mapper.map(indexes[position], values[position], message);
			}
			return this;
		}
		default: {
			EnvironmentContext context = EnvironmentContext.getContext();
			MessageStorage storage = MathCalculator.getStorage();
			Semaphore semaphore = storage.getSemaphore();
			for (int position = 0; position < size; position++) {
				int index = position;
				context.doStructureByAny(position, () -> {
					T copy = storage.detachMessage(message);
					values[index] = mapper.map(indexes[index], values[index], copy);
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

	public <T extends MathMessage> void modifyIndexes(VectorMapper<T> mapper, T message, int... indices) {
		assert indexes.length >= indices.length;
		int current = Integer.MIN_VALUE;
		int position = 0;
		for (int index : indices) {
			if (current >= index) {
				throw new IllegalArgumentException();
			}
			current = index;
			if (current < 0) {
				throw new IllegalArgumentException();
			}
			indexes[position] = indices[position];
			values[position] = mapper.map(indices[position], 0F, message);
			position++;
		}
		int oldElementSize = size;
		int oldKnownSize = getKnownSize();
		int oldUnknownSize = getUnknownSize();
		size = indices.length;
		int newElementSize = size;
		int newKnownSize = getKnownSize();
		int newUnknownSize = getUnknownSize();
		for (MathMonitor<VectorScalar> monitor : monitors.keySet()) {
			monitor.notifySizeChanged(this, oldElementSize, newElementSize, oldKnownSize, newKnownSize, oldUnknownSize, newUnknownSize);
		}
	}

	@Override
	public boolean equals(Object object) {
		if (this == object)
			return true;
		if (object == null)
			return false;
		if (getClass() != object.getClass())
			return false;
		ArrayVector that = (ArrayVector) object;
		EqualsBuilder equal = new EqualsBuilder();
		equal.append(this.capacity, that.capacity);
		equal.append(this.size, that.size);
		equal.append(this.indexes, that.indexes);
		equal.append(this.values, that.values);
		return equal.isEquals();
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(capacity);
		hash.append(size);
		hash.append(indexes);
		hash.append(values);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		StringBuilder buffer = new StringBuilder();
		for (int index = 0; index < size; index++) {
			buffer.append(values[index]).append(", ");
		}
		buffer.append("\n");
		return buffer.toString();
	}

	@Override
	public Iterator<VectorScalar> iterator() {
		return new ArrayVectorIterator();
	}

	private class ArrayVectorIterator implements Iterator<VectorScalar> {

		private int cursor = 0;

		private final ArrayVectorScalar term = new ArrayVectorScalar();

		@Override
		public boolean hasNext() {
			return cursor < size;
		}

		@Override
		public VectorScalar next() {
			term.update(cursor++);
			return term;
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}

	}

	private class ArrayVectorScalar implements VectorScalar {

		private int cursor;

		private void update(int cursor) {
			this.cursor = cursor;
		}

		@Override
		public int getIndex() {
			return indexes[cursor];
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

	ArrayVector() {
	}

	public ArrayVector(int capacity, int[] indexes, float[] values) {
		assert indexes.length == values.length;
		this.capacity = capacity;
		this.size = indexes.length;
		assert capacity >= size;
		this.indexes = indexes;
		this.values = values;
	}

	public ArrayVector(int capacity, int[] indexes, VectorMapper<?> mapper) {
		this.capacity = capacity;
		this.size = indexes.length;
		assert capacity >= size;
		this.indexes = indexes;
		this.values = new float[size];
		for (int index = 0; index < size; index++) {
			this.values[index] = mapper.map(index, this.values[index], null);
		}
	}

	public ArrayVector(SparseVector vector) {
		this.capacity = vector.getKnownSize() + vector.getUnknownSize();
		this.size = vector.getElementSize();
		this.indexes = new int[size];
		this.values = new float[size];
		int index = 0;
		for (VectorScalar term : vector) {
			this.indexes[index] = term.getIndex();
			this.values[index] = term.getValue();
			index++;
		}
	}

	public ArrayVector(SparseVector vector, VectorMapper<?> mapper) {
		this.capacity = vector.getKnownSize() + vector.getUnknownSize();
		this.size = vector.getElementSize();
		this.indexes = new int[size];
		this.values = new float[size];
		int index = 0;
		for (VectorScalar term : vector) {
			this.indexes[index] = term.getIndex();
			this.values[index] = mapper.map(term.getIndex(), term.getValue(), null);
			index++;
		}
	}

}
