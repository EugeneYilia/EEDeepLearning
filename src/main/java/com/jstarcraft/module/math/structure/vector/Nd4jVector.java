package com.jstarcraft.module.math.structure.vector;

import java.util.Iterator;
import java.util.concurrent.Semaphore;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.concurrency.AffinityManager.Location;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.environment.EnvironmentThread;
import com.jstarcraft.module.math.structure.MathAccessor;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MathIterator;
import com.jstarcraft.module.math.structure.MathMessage;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.Nd4jMatrix;
import com.jstarcraft.module.math.structure.message.MessageStorage;
import com.jstarcraft.module.model.ModelCycle;
import com.jstarcraft.module.model.ModelDefinition;

@ModelDefinition(value = { "size", "order", "data" })
public class Nd4jVector implements MathVector, ModelCycle {

	private static final AffinityManager manager = Nd4j.getAffinityManager();

	private static final Double one = 1D;

	private static final double zero = 0D;

	private int size;

	private char order;

	private float[] data;

	private INDArray vector;

	Nd4jVector() {
	}

	public Nd4jVector(INDArray vector) {
		if (vector.rank() != 1) {
			new IllegalArgumentException();
		}
		this.size = vector.length();
		this.order = vector.ordering();
		this.vector = vector;
	}

	@Override
	public int getElementSize() {
		return vector.length();
	}

	@Override
	public int getKnownSize() {
		return vector.length();
	}

	@Override
	public int getUnknownSize() {
		return 0;
	}

	@Override
	public MathIterator<VectorScalar> iterateElement(MathCalculator mode, MathAccessor<VectorScalar>... accessors) {
		int size = vector.length();
		switch (mode) {
		case SERIAL: {
			Nd4jVectorScalar scalar = new Nd4jVectorScalar();
			for (int index = 0; index < size; index++) {
				scalar.update(index);
				for (MathAccessor<VectorScalar> accessor : accessors) {
					accessor.accessScalar(scalar);
				}
			}
			return this;
		}
		default: {
			// TODO 参考Nd4jMatrix性能优化
			EnvironmentContext context = EnvironmentContext.getContext();
			MessageStorage storage = MathCalculator.getStorage();
			Semaphore semaphore = storage.getSemaphore();
			for (int index = 0; index < size; index++) {
				int elementIndex = index;
				context.doStructureByAny(index, () -> {
					Nd4jVectorScalar scalar = new Nd4jVectorScalar();
					scalar.update(elementIndex);
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
	public MathIterator<VectorScalar> setValues(float value) {
		vector.assign(value);
		return this;
	}

	@Override
	public MathIterator<VectorScalar> scaleValues(float value) {
		vector.muli(value);
		return this;
	}

	@Override
	public MathIterator<VectorScalar> shiftValues(float value) {
		vector.addi(value);
		return this;
	}

	@Override
	public float getSum(boolean absolute) {
		if (absolute) {
			return vector.ameanNumber().floatValue() * vector.length();
		} else {
			return vector.sumNumber().floatValue();
		}
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
		return vector.getFloat(position);
	}

	@Override
	public void setValue(int position, float value) {
		vector.putScalar(position, value);
	}

	@Override
	public void scaleValue(int position, float value) {
		vector.putScalar(position, vector.getFloat(position) * value);
	}

	@Override
	public void shiftValue(int position, float value) {
		vector.putScalar(position, vector.getFloat(position) + value);
	}

	@Override
	public <T extends MathMessage> Nd4jVector collectValues(VectorCollector<T> collector, T message, MathCalculator mode) {
		int size = vector.length();
		switch (mode) {
		case SERIAL: {
			for (int index = 0; index < size; index++) {
				collector.collect(index, vector.getFloat(index), message);
			}
			return this;
		}
		default: {
			// TODO 参考Nd4jMatrix性能优化
			EnvironmentContext context = EnvironmentContext.getContext();
			MessageStorage storage = MathCalculator.getStorage();
			Semaphore semaphore = storage.getSemaphore();
			for (int index = 0; index < size; index++) {
				int elementIndex = index;
				context.doStructureByAny(index, () -> {
					T copy = storage.detachMessage(message);
					collector.collect(elementIndex, vector.getFloat(elementIndex), copy);
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
	public <T extends MathMessage> Nd4jVector mapValues(VectorMapper<T> mapper, T message, MathCalculator mode) {
		int size = vector.length();
		switch (mode) {
		case SERIAL: {
			for (int index = 0; index < size; index++) {
				vector.putScalar(index, mapper.map(index, vector.getFloat(index), message));
			}
			return this;
		}
		default: {
			// TODO 参考Nd4jMatrix性能优化
			EnvironmentContext context = EnvironmentContext.getContext();
			MessageStorage storage = MathCalculator.getStorage();
			Semaphore semaphore = storage.getSemaphore();
			for (int index = 0; index < size; index++) {
				int elementIndex = index;
				context.doStructureByAny(index, () -> {
					T copy = storage.detachMessage(message);
					vector.putScalar(elementIndex, mapper.map(elementIndex, vector.getFloat(elementIndex), copy));
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

	// @Override
	// public <T extends DataMessage> Nd4jVector calculate(VectorCollector<T>
	// collector, T message, Calculator mode) {
	// // 保证内存与显存同步
	// manager.ensureLocation(vector, Location.HOST);
	// manager.tagLocation(vector, Location.HOST);
	// EnvironmentThread thread = EnvironmentThread.currentThread();
	// data = thread.getArray();
	// // TODO 此处存在Bug,pointer可能会指向矩阵,需要仔细考虑与测试各种情况.
	// FloatPointer pointer = (FloatPointer) vector.data().pointer();
	// int size = vector.length();
	// pointer.get(data, 0, size);
	// switch (mode) {
	// case SERIAL: {
	// for (int index = 0; index < size; index++) {
	// collector.collect(index, data[index], message);
	// }
	// return this;
	// }
	// default: {
	// EnvironmentContext context = EnvironmentContext.getContext();
	// MessageStorage storage = Calculator.getStorage();
	// Semaphore semaphore = storage.getSemaphore();
	// for (int index = 0; index < size; index++) {
	// int elementIndex = index;
	// context.doStructureByAny(() -> {
	// T copy = storage.detachMessage(message);
	// collector.collect(elementIndex, data[elementIndex], copy);
	// semaphore.release();
	// });
	// }
	// try {
	// semaphore.acquire(size);
	// storage.attachMessage(message);
	// } catch (Exception exception) {
	// throw new RuntimeException(exception);
	// }
	// return this;
	// }
	// }
	// }
	//
	// @Override
	// public <T extends DataMessage> Nd4jVector calculate(VectorMapper<T> mapper, T
	// message, Calculator mode) {
	// // 保证内存与显存同步
	// manager.ensureLocation(vector, Location.HOST);
	// manager.tagLocation(vector, Location.HOST);
	// EnvironmentThread thread = EnvironmentThread.currentThread();
	// data = thread.getArray();
	// // TODO 此处存在Bug,pointer可能会指向矩阵,需要仔细考虑与测试各种情况.
	// FloatPointer pointer = (FloatPointer) vector.data().pointer();
	// int size = vector.length();
	// pointer.get(data, 0, size);
	// switch (mode) {
	// case SERIAL: {
	// for (int index = 0; index < size; index++) {
	// data[index] = (float) mapper.map(index, data[index], message);
	// }
	// pointer.put(data, 0, size);
	// return this;
	// }
	// default: {
	// // TODO 参考Nd4jMatrix性能优化
	// EnvironmentContext context = EnvironmentContext.getContext();
	// MessageStorage storage = Calculator.getStorage();
	// Semaphore semaphore = storage.getSemaphore();
	// for (int index = 0; index < size; index++) {
	// int elementIndex = index;
	// context.doStructureByAny(() -> {
	// T copy = storage.detachMessage(message);
	// data[elementIndex] = (float) mapper.map(elementIndex, data[elementIndex],
	// copy);
	// semaphore.release();
	// });
	// }
	// try {
	// semaphore.acquire(size);
	// storage.attachMessage(message);
	// } catch (Exception exception) {
	// throw new RuntimeException(exception);
	// }
	// pointer.put(data, 0, size);
	// return this;
	// }
	// }
	// }

	@Override
	public MathVector addVector(MathVector vector) {
		if (vector instanceof Nd4jVector) {
			INDArray dataArray = this.getArray();
			// TODO 此处可能需要修改方向.
			INDArray vectorArray = Nd4jVector.class.cast(vector).getArray();
			dataArray.addi(vectorArray);
			return this;
		} else {
			return MathVector.super.addVector(vector);
		}
	}

	@Override
	public MathVector subtractVector(MathVector vector) {
		if (vector instanceof Nd4jVector) {
			INDArray dataArray = this.getArray();
			// TODO 此处可能需要修改方向.
			INDArray vectorArray = Nd4jVector.class.cast(vector).getArray();
			dataArray.subi(vectorArray);
			return this;
		} else {
			return MathVector.super.addVector(vector);
		}
	}

	@Override
	public MathVector multiplyVector(MathVector vector) {
		if (vector instanceof Nd4jVector) {
			INDArray dataArray = this.getArray();
			// TODO 此处可能需要修改方向.
			INDArray vectorArray = Nd4jVector.class.cast(vector).getArray();
			dataArray.muli(vectorArray);
			return this;
		} else {
			return MathVector.super.addVector(vector);
		}
	}

	@Override
	public MathVector divideVector(MathVector vector) {
		if (vector instanceof Nd4jVector) {
			INDArray dataArray = this.getArray();
			// TODO 此处可能需要修改方向.
			INDArray vectorArray = Nd4jVector.class.cast(vector).getArray();
			dataArray.divi(vectorArray);
			return this;
		} else {
			return MathVector.super.addVector(vector);
		}
	}

	@Override
	public MathVector copyVector(MathVector vector) {
		if (vector instanceof Nd4jVector) {
			INDArray dataArray = this.getArray();
			// TODO 此处可能需要修改方向.
			INDArray vectorArray = Nd4jVector.class.cast(vector).getArray();
			dataArray.assign(vectorArray);
			return this;
		} else {
			return MathVector.super.addVector(vector);
		}
	}

	@Override
	public MathVector dotProduct(MathMatrix leftMatrix, boolean transpose, MathVector rightVector, MathCalculator mode) {
		if (leftMatrix instanceof Nd4jMatrix && rightVector instanceof Nd4jVector) {
			EnvironmentThread thread = EnvironmentThread.currentThread();
			try (MemoryWorkspace workspace = thread.getSpace()) {
				INDArray leftArray = transpose ? Nd4jMatrix.class.cast(leftMatrix).getArray().transpose() : Nd4jMatrix.class.cast(leftMatrix).getArray();
				INDArray rightArray = Nd4jVector.class.cast(rightVector).getArray();
				INDArray dataArray = this.getArray();
				Nd4j.getBlasWrapper().gemv(one, leftArray, rightArray, zero, dataArray);
				return this;
			}
		} else {
			return MathVector.super.dotProduct(leftMatrix, transpose, rightVector, mode);
		}
	}

	@Override
	public MathVector dotProduct(MathVector leftVector, MathMatrix rightMatrix, boolean transpose, MathCalculator mode) {
		if (leftVector instanceof Nd4jVector && rightMatrix instanceof Nd4jMatrix) {
			EnvironmentThread thread = EnvironmentThread.currentThread();
			try (MemoryWorkspace workspace = thread.getSpace()) {
				INDArray leftArray = Nd4jVector.class.cast(leftVector).getArray();
				if (leftArray.isView()) {
					// 此处执行复制是由于gemm不支持视图向量.
					leftArray = leftArray.dup();
				}
				if (leftArray.columns() == 1) {
					leftArray = leftArray.transpose();
				}
				INDArray rightArray = transpose ? Nd4jMatrix.class.cast(rightMatrix).getArray().transpose() : Nd4jMatrix.class.cast(rightMatrix).getArray();
				INDArray dataArray = this.getArray();
				leftArray.mmul(rightArray, dataArray);
				// Nd4j.getBlasWrapper().level3().gemm(leftArray, rightArray, dataArray, false,
				// false, one, zero);
				return this;
			}
		} else {
			return MathVector.super.dotProduct(leftVector, rightMatrix, transpose, mode);
		}
	}

	@Override
	public MathVector accumulateProduct(MathMatrix leftMatrix, boolean transpose, MathVector rightVector, MathCalculator mode) {
		if (leftMatrix instanceof Nd4jMatrix && rightVector instanceof Nd4jVector) {
			EnvironmentThread thread = EnvironmentThread.currentThread();
			try (MemoryWorkspace workspace = thread.getSpace()) {
				INDArray leftArray = transpose ? Nd4jMatrix.class.cast(leftMatrix).getArray().transpose() : Nd4jMatrix.class.cast(leftMatrix).getArray();
				INDArray rightArray = Nd4jVector.class.cast(rightVector).getArray();
				INDArray dataArray = this.getArray();
				INDArray cacheArray = Nd4j.zeros(dataArray.shape(), dataArray.ordering());
				Nd4j.getBlasWrapper().gemv(one, leftArray, rightArray, zero, cacheArray);
				dataArray.addi(cacheArray);
				return this;
			}
		} else {
			return MathVector.super.accumulateProduct(leftMatrix, transpose, rightVector, mode);
		}
	}

	@Override
	public MathVector accumulateProduct(MathVector leftVector, MathMatrix rightMatrix, boolean transpose, MathCalculator mode) {
		if (leftVector instanceof Nd4jVector && rightMatrix instanceof Nd4jMatrix) {
			EnvironmentThread thread = EnvironmentThread.currentThread();
			try (MemoryWorkspace workspace = thread.getSpace()) {
				INDArray leftArray = Nd4jVector.class.cast(leftVector).getArray();
				if (leftArray.isView()) {
					// 此处执行复制是由于gemm不支持视图向量.
					leftArray = leftArray.dup();
				}
				if (leftArray.columns() == 1) {
					leftArray = leftArray.transpose();
				}
				INDArray rightArray = transpose ? Nd4jMatrix.class.cast(rightMatrix).getArray().transpose() : Nd4jMatrix.class.cast(rightMatrix).getArray();
				INDArray dataArray = this.getArray();
				INDArray cacheArray = Nd4j.zeros(dataArray.shape(), dataArray.ordering());
				leftArray.mmul(rightArray, cacheArray);
				dataArray.addi(cacheArray);
				// Nd4j.getBlasWrapper().level3().gemm(leftArray, rightArray, dataArray, false,
				// false, one, zero);
				return this;
			}
		} else {
			return MathVector.super.accumulateProduct(leftVector, rightMatrix, transpose, mode);
		}
	}

	public INDArray getArray() {
		return vector;
	}

	// @Deprecated
	// // TODO 考虑使用Worksapce代替
	// public INDArray getRowCache() {
	// if (rowCache == null) {
	// rowCache = vector;
	// if (rowCache.isView()) {
	// // 此处执行复制是由于gemm不支持视图向量.
	// rowCache = rowCache.dup();
	// }
	// if (rowCache.columns() == 1) {
	// rowCache = rowCache.transpose();
	// }
	// }
	// return rowCache;
	// }
	//
	// @Deprecated
	// // TODO 考虑使用Worksapce代替
	// public INDArray getColumnCache() {
	// if (columnCache == null) {
	// columnCache = vector;
	// if (columnCache.isView()) {
	// // 此处执行复制是由于gemm不支持视图向量.
	// columnCache = columnCache.dup();
	// }
	// if (columnCache.rows() == 1) {
	// columnCache = columnCache.transpose();
	// }
	// }
	// return columnCache;
	// }

	@Override
	public void beforeSave() {
		data = new float[size];
		FloatPointer pointer = (FloatPointer) vector.data().pointer();
		pointer.get(data, 0, data.length);
	}

	@Override
	public void afterLoad() {
		vector = Nd4j.zeros(size, order);
		manager.ensureLocation(vector, Location.HOST);
		manager.tagLocation(vector, Location.HOST);
		FloatPointer pointer = (FloatPointer) vector.data().pointer();
		pointer.put(data, 0, data.length);
		data = null;
	}

	@Override
	public boolean equals(Object object) {
		if (this == object)
			return true;
		if (object == null)
			return false;
		if (getClass() != object.getClass())
			return false;
		Nd4jVector that = (Nd4jVector) object;
		EqualsBuilder equal = new EqualsBuilder();
		equal.append(this.vector, that.vector);
		return equal.isEquals();
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(vector);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return vector.toString();
	}

	@Override
	public Iterator<VectorScalar> iterator() {
		return new Nd4jVectorIterator();
	}

	private class Nd4jVectorIterator implements Iterator<VectorScalar> {

		private int index = 0, size = vector.length();

		private final Nd4jVectorScalar term = new Nd4jVectorScalar();

		@Override
		public boolean hasNext() {
			return index < size;
		}

		@Override
		public VectorScalar next() {
			term.update(index++);
			return term;
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}

	}

	private class Nd4jVectorScalar implements VectorScalar {

		private int index;

		private void update(int index) {
			this.index = index;
		}

		@Override
		public int getIndex() {
			return index;
		}

		@Override
		public float getValue() {
			return vector.getFloat(index);
		}

		@Override
		public void scaleValue(float value) {
			vector.putScalar(index, vector.getFloat(index) * value);
		}

		@Override
		public void setValue(float value) {
			vector.putScalar(index, value);
		}

		@Override
		public void shiftValue(float value) {
			vector.putScalar(index, vector.getFloat(index) + value);
		}

	}

}
