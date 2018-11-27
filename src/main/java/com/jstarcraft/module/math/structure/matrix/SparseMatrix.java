package com.jstarcraft.module.math.structure.matrix;

import java.util.Arrays;
import java.util.Iterator;
import java.util.concurrent.Semaphore;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.math3.util.FastMath;

import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;
import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.structure.MathAccessor;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MathIterator;
import com.jstarcraft.module.math.structure.MathMessage;
import com.jstarcraft.module.math.structure.message.MessageStorage;
import com.jstarcraft.module.math.structure.vector.SparseVector;

/**
 * 稀疏矩阵
 * 
 * <pre>
 * 稀疏矩阵的使用规约:
 * 1.稀疏矩阵整个生命周期数量与坐标保持不变.
 * 2.无论怎么操作,应该保证矩阵中的元素几乎应该都是非0.
 * 3.尽量在准备阶段构建稀疏矩阵.
 * 如果需要动态设置的矩阵类型,参考{@link DenseMatrix} 或者 {@link Nd4jMatrix}
 * 
 * 稀疏矩阵的存储格式:https://software.intel.com/en-us/node/471374
 * </pre>
 * 
 * @author Birdy
 *
 */
public class SparseMatrix implements MathMatrix {

	/** 行列大小 */
	private int rowSize, columnSize;

	// Compressed Row Storage (CRS)
	private int[] rowPoints;
	private int[] rowIndexes;

	// Compressed Column Storage (CCS)
	private int[] columnPoints;
	private int[] columnIndexes;

	private int[] termRows;
	private int[] termColumns;
	private float[] termValues;

	private SparseMatrix() {
	}

	@Override
	public int getElementSize() {
		return termValues.length;
	}

	@Override
	public int getKnownSize() {
		return getElementSize();
	}

	@Override
	public int getUnknownSize() {
		return rowSize * columnSize - getElementSize();
	}
	
	@Override
	public MathIterator<MatrixScalar> iterateElement(MathCalculator mode, MathAccessor<MatrixScalar>... accessors) {
		switch (mode) {
		case SERIAL: {
			SparseMatrixScalar scalar = new SparseMatrixScalar();
			for (int cursor = 0, size = termValues.length; cursor < size; cursor++) {
				scalar.update(cursor);
				for (MathAccessor<MatrixScalar> accessor : accessors) {
					accessor.accessScalar(scalar);
				}
			}
			return this;
		}
		default: {
			if (this.getColumnSize() <= this.getRowSize()) {
				int size = columnSize;
				EnvironmentContext context = EnvironmentContext.getContext();
				MessageStorage storage = MathCalculator.getStorage();
				Semaphore semaphore = storage.getSemaphore();
				for (int index = 0; index < size; index++) {
					int columnIndex = index;
					context.doStructureByAny(index, () -> {
						SparseMatrixScalar scalar = new SparseMatrixScalar();
						int beginPoint = columnPoints[columnIndex], endPoint = columnPoints[columnIndex + 1];
						for (; beginPoint < endPoint; beginPoint++) {
							int cursor = columnIndexes[beginPoint];
							scalar.update(cursor);
							for (MathAccessor<MatrixScalar> accessor : accessors) {
								accessor.accessScalar(scalar);
							}
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
			} else {
				int size = rowSize;
				EnvironmentContext context = EnvironmentContext.getContext();
				MessageStorage storage = MathCalculator.getStorage();
				Semaphore semaphore = storage.getSemaphore();
				for (int index = 0; index < size; index++) {
					int rowIndex = index;
					context.doStructureByAny(index, () -> {
						SparseMatrixScalar scalar = new SparseMatrixScalar();
						int beginPoint = rowPoints[rowIndex], endPoint = rowPoints[rowIndex + 1];
						for (; beginPoint < endPoint; beginPoint++) {
							int cursor = rowIndexes[beginPoint];
							scalar.update(cursor);
							for (MathAccessor<MatrixScalar> accessor : accessors) {
								accessor.accessScalar(scalar);
							}
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
	}

	@Override
	public SparseMatrix setValues(float value) {
		for (int cursor = 0, size = termValues.length; cursor < size; cursor++) {
			termValues[cursor] = value;
		}
		return this;
	}

	@Override
	public SparseMatrix scaleValues(float value) {
		for (int cursor = 0, size = termValues.length; cursor < size; cursor++) {
			termValues[cursor] *= value;
		}
		return this;
	}

	@Override
	public SparseMatrix shiftValues(float value) {
		for (int cursor = 0, size = termValues.length; cursor < size; cursor++) {
			termValues[cursor] += value;
		}
		return this;
	}

	@Override
	public float getSum(boolean absolute) {
		float sum = 0F;
		if (absolute) {
			for (float value : termValues) {
				sum += FastMath.abs(value);
			}
		} else {
			for (float value : termValues) {
				sum += value;
			}
		}
		return sum;
	}

	@Override
	public int getRowSize() {
		return rowSize;
	}

	@Override
	public int getColumnSize() {
		return columnSize;
	}

	@Override
	public SparseVector getRowVector(int rowIndex) {
		SparseVector vector = new SparseVector(columnSize, rowIndexes, termColumns, termValues, rowPoints[rowIndex], rowPoints[rowIndex + 1]);
		return vector;
	}

	@Override
	public SparseVector getColumnVector(int columnIndex) {
		SparseVector vector = new SparseVector(rowSize, columnIndexes, termRows, termValues, columnPoints[columnIndex], columnPoints[columnIndex + 1]);
		return vector;
	}

	@Override
	public boolean isIndexed() {
		return false;
	}

	@Override
	public float getValue(int rowIndex, int columnIndex) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void setValue(int rowIndex, int columnIndex, float value) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void scaleValue(int rowIndex, int columnIndex, float value) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void shiftValue(int rowIndex, int columnIndex, float value) {
		throw new UnsupportedOperationException();
	}

	@Override
	public <T extends MathMessage> SparseMatrix collectValues(MatrixCollector<T> collector, T message, MathCalculator mode) {
		switch (mode) {
		case SERIAL: {
			for (int cursor = 0, size = termValues.length; cursor < size; cursor++) {
				collector.collect(termRows[cursor], termColumns[cursor], termValues[cursor], message);
			}
			return this;
		}
		default: {
			if (this.getColumnSize() <= this.getRowSize()) {
				int size = columnSize;
				EnvironmentContext context = EnvironmentContext.getContext();
				MessageStorage storage = MathCalculator.getStorage();
				Semaphore semaphore = storage.getSemaphore();
				for (int index = 0; index < size; index++) {
					int columnIndex = index;
					context.doStructureByAny(index, () -> {
						T copy = storage.detachMessage(message);
						int beginPoint = columnPoints[columnIndex], endPoint = columnPoints[columnIndex + 1];
						for (; beginPoint < endPoint; beginPoint++) {
							int cursor = columnIndexes[beginPoint];
							collector.collect(termRows[cursor], termColumns[cursor], termValues[cursor], copy);
						}
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
			} else {
				int size = rowSize;
				EnvironmentContext context = EnvironmentContext.getContext();
				MessageStorage storage = MathCalculator.getStorage();
				Semaphore semaphore = storage.getSemaphore();
				for (int index = 0; index < size; index++) {
					int rowIndex = index;
					context.doStructureByAny(index, () -> {
						T copy = storage.detachMessage(message);
						int beginPoint = rowPoints[rowIndex], endPoint = rowPoints[rowIndex + 1];
						for (; beginPoint < endPoint; beginPoint++) {
							int cursor = rowIndexes[beginPoint];
							collector.collect(termRows[cursor], termColumns[cursor], termValues[cursor], copy);
						}
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
	}

	@Override
	public <T extends MathMessage> SparseMatrix mapValues(MatrixMapper<T> mapper, T message, MathCalculator mode) {
		switch (mode) {
		case SERIAL: {
			for (int cursor = 0, size = termValues.length; cursor < size; cursor++) {
				termValues[cursor] = mapper.map(termRows[cursor], termColumns[cursor], termValues[cursor], message);
			}
			return this;
		}
		default: {
			if (this.getColumnSize() <= this.getRowSize()) {
				int size = columnSize;
				EnvironmentContext context = EnvironmentContext.getContext();
				MessageStorage storage = MathCalculator.getStorage();
				Semaphore semaphore = storage.getSemaphore();
				for (int index = 0; index < size; index++) {
					int columnIndex = index;
					context.doStructureByAny(index, () -> {
						T copy = storage.detachMessage(message);
						int beginPoint = columnPoints[columnIndex], endPoint = columnPoints[columnIndex + 1];
						for (; beginPoint < endPoint; beginPoint++) {
							int cursor = columnIndexes[beginPoint];
							termValues[cursor] = mapper.map(termRows[cursor], termColumns[cursor], termValues[cursor], copy);
						}
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
			} else {
				int size = rowSize;
				EnvironmentContext context = EnvironmentContext.getContext();
				MessageStorage storage = MathCalculator.getStorage();
				Semaphore semaphore = storage.getSemaphore();
				for (int index = 0; index < size; index++) {
					int rowIndex = index;
					context.doStructureByAny(index, () -> {
						T copy = storage.detachMessage(message);
						int beginPoint = rowPoints[rowIndex], endPoint = rowPoints[rowIndex + 1];
						for (; beginPoint < endPoint; beginPoint++) {
							int cursor = rowIndexes[beginPoint];
							termValues[cursor] = mapper.map(termRows[cursor], termColumns[cursor], termValues[cursor], copy);
						}
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
	}

	public int getRowScope(int rowIndex) {
		return rowPoints[rowIndex + 1] - rowPoints[rowIndex];
	}

	public int getColumnScope(int columnIndex) {
		return columnPoints[columnIndex + 1] - columnPoints[columnIndex];
	}

	// TODO 此方法准备取消,由StructureUtility.normalize代替.
	@Deprecated
	public void normalize(float minimum, float maximum) {
		// TODO 考虑将其移到HashMatrix
		assert maximum > minimum;
		for (int index = 0; index < termValues.length; index++) {
			termValues[index] = (termValues[index] - minimum) / (maximum - minimum);
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
		SparseMatrix that = (SparseMatrix) object;
		EqualsBuilder equal = new EqualsBuilder();
		equal.append(this.rowSize, that.rowSize);
		equal.append(this.columnSize, that.columnSize);
		equal.append(this.termValues, that.termValues);
		return equal.isEquals();
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(rowSize);
		hash.append(columnSize);
		hash.append(termValues);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		ToStringBuilder string = new ToStringBuilder(this);
		string.append(rowSize);
		string.append(columnSize);
		string.append(getElementSize());
		return string.toString();
	}

	@Override
	public Iterator<MatrixScalar> iterator() {
		return new SparseMatrixIterator();
	}

	private class SparseMatrixIterator implements Iterator<MatrixScalar> {

		private int cursor;

		private SparseMatrixScalar term = new SparseMatrixScalar();

		@Override
		public boolean hasNext() {
			return cursor < termValues.length;
		}

		@Override
		public MatrixScalar next() {
			term.update(cursor++);
			return term;
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}

	}

	private class SparseMatrixScalar implements MatrixScalar {

		private int index;

		private void update(int index) {
			this.index = index;
		}

		@Override
		public int getRow() {
			return termRows[index];
		}

		@Override
		public int getColumn() {
			return termColumns[index];
		}

		@Override
		public float getValue() {
			return termValues[index];
		}

		@Override
		public void scaleValue(float value) {
			termValues[index] *= value;
		}

		@Override
		public void setValue(float value) {
			termValues[index] = value;
		}

		@Override
		public void shiftValue(float value) {
			termValues[index] += value;
		}

	}

	private void copyCRS(int[] pointers, int[] indexes) {
		rowPoints = pointers;
		rowIndexes = indexes;
	}

	private void copyCCS(int[] pointers, int[] indexes) {
		columnPoints = pointers;
		columnIndexes = indexes;
	}

	public static SparseMatrix copyOf(SparseMatrix matrix, boolean transpose, MatrixMapper<?> mapper) {
		SparseMatrix instance = new SparseMatrix();
		if (transpose) {
			instance.rowSize = matrix.columnSize;
			instance.columnSize = matrix.rowSize;
			instance.copyCRS(matrix.columnPoints, matrix.columnIndexes);
			instance.copyCCS(matrix.rowPoints, matrix.rowIndexes);
			instance.termRows = matrix.termColumns;
			instance.termColumns = matrix.termRows;
		} else {
			instance.rowSize = matrix.rowSize;
			instance.columnSize = matrix.columnSize;
			instance.copyCRS(matrix.rowPoints, matrix.rowIndexes);
			instance.copyCCS(matrix.columnPoints, matrix.columnIndexes);
			instance.termRows = matrix.termRows;
			instance.termColumns = matrix.termColumns;
		}
		instance.termValues = Arrays.copyOf(matrix.termValues, matrix.termValues.length);
		for (MatrixScalar term : instance) {
			float value = mapper.map(term.getRow(), term.getColumn(), term.getValue(), null);
			term.setValue(value);
		}
		return instance;
	}

	public static SparseMatrix valueOf(int rowSize, int columnSize, Table<Integer, Integer, Float> dataTable) {
		SparseMatrix instance = new SparseMatrix();
		instance.rowSize = rowSize;
		instance.columnSize = columnSize;
		int size = dataTable.size();

		// CRS
		instance.rowPoints = new int[rowSize + 1];
		instance.rowIndexes = new int[size];

		// CCS
		instance.columnPoints = new int[columnSize + 1];
		instance.columnIndexes = new int[size];

		instance.termRows = new int[size];
		instance.termColumns = new int[size];
		instance.termValues = new float[size];

		int[] rowCounts = new int[rowSize];
		int[] columnCounts = new int[columnSize];
		Integer[] rowIndexes = new Integer[size];
		Integer[] columnIndexes = new Integer[size];
		int index = 0;
		for (Cell<Integer, Integer, Float> cell : dataTable.cellSet()) {
			int row = cell.getRowKey();
			int column = cell.getColumnKey();
			// 设置term的坐标与值
			instance.termRows[index] = row;
			instance.termColumns[index] = column;
			instance.termValues[index] = cell.getValue();
			// 统计行列的大小
			rowCounts[row]++;
			columnCounts[column]++;
			index++;
		}

		for (int point = 1; point <= rowSize; ++point) {
			// 设置行指针
			int row = point - 1;
			instance.rowPoints[point] = instance.rowPoints[row] + rowCounts[row];
		}

		for (int point = 1; point <= columnSize; ++point) {
			// 设置列指针
			int column = point - 1;
			instance.columnPoints[point] = instance.columnPoints[column] + columnCounts[column];
		}

		for (index = 0; index < size; index++) {
			// 设置行列的索引
			int row = instance.termRows[index];
			int column = instance.termColumns[index];
			rowIndexes[index] = instance.rowPoints[row] + (--rowCounts[row]);
			columnIndexes[index] = instance.columnPoints[column] + (--columnCounts[column]);
		}

		// 排序行的索引
		Arrays.sort(rowIndexes, (left, right) -> {
			int value = instance.termRows[left] - instance.termRows[right];
			if (value == 0) {
				value = instance.termColumns[left] - instance.termColumns[right];
			}
			return value;
		});

		// 排序列的索引
		Arrays.sort(columnIndexes, (left, right) -> {
			int value = instance.termColumns[left] - instance.termColumns[right];
			if (value == 0) {
				value = instance.termRows[left] - instance.termRows[right];
			}
			return value;
		});

		for (index = 0; index < size; index++) {
			// 拷贝行列的索引
			instance.rowIndexes[index] = rowIndexes[index];
			instance.columnIndexes[index] = columnIndexes[index];
		}

		return instance;
	}

}
