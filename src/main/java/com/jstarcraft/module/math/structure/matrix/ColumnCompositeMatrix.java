package com.jstarcraft.module.math.structure.matrix;

import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.concurrent.Semaphore;

import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.structure.MathAccessor;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MathIterator;
import com.jstarcraft.module.math.structure.MathMessage;
import com.jstarcraft.module.math.structure.message.MessageStorage;
import com.jstarcraft.module.math.structure.vector.CompositeVector;
import com.jstarcraft.module.math.structure.vector.MathVector;

public class ColumnCompositeMatrix extends CompositeMatrix {

	@Override
	public MathIterator<MatrixScalar> iterateElement(MathCalculator mode, MathAccessor<MatrixScalar>... accessors) {
		switch (mode) {
		case SERIAL: {
			CompositeMatrixScalar scalar = new CompositeMatrixScalar();
			int size = components.length;
			for (int point = 0; point < size; point++) {
				MathMatrix matrix = components[point];
				int split = splits[point];
				for (MatrixScalar term : matrix) {
					scalar.update(term, term.getRow(), term.getColumn() + split);
					for (MathAccessor<MatrixScalar> accessor : accessors) {
						accessor.accessScalar(scalar);
					}
				}
			}
			return this;
		}
		default: {
			EnvironmentContext context = EnvironmentContext.getContext();
			MessageStorage storage = MathCalculator.getStorage();
			Semaphore semaphore = storage.getSemaphore();
			int size = components.length;
			for (int point = 0; point < size; point++) {
				MathMatrix matrix = components[point];
				int split = splits[point];
				context.doStructureByAny(point, () -> {
					CompositeMatrixScalar scalar = new CompositeMatrixScalar();
					for (MatrixScalar term : matrix) {
						scalar.update(term, term.getRow(), term.getColumn() + split);
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

	@Override
	public MathVector getRowVector(int rowIndex) {
		MathVector[] vectors = new MathVector[components.length];
		for (int point = 0, size = components.length; point < size; point++) {
			vectors[point] = components[point].getRowVector(rowIndex);
		}
		return CompositeVector.valueOf(indexed ? points : null, vectors);
	}

	@Override
	public MathVector getColumnVector(int columnIndex) {
		int point = points[columnIndex];
		return components[point].getColumnVector(columnIndex - splits[point]);
	}

	@Override
	public float getValue(int rowIndex, int columnIndex) {
		int point = points[columnIndex];
		return components[point].getValue(rowIndex, columnIndex - splits[point]);
	}

	@Override
	public void setValue(int rowIndex, int columnIndex, float value) {
		int point = points[columnIndex];
		components[point].setValue(rowIndex, columnIndex - splits[point], value);
	}

	@Override
	public void scaleValue(int rowIndex, int columnIndex, float value) {
		int point = points[columnIndex];
		components[point].scaleValue(rowIndex, columnIndex - splits[point], value);
	}

	@Override
	public void shiftValue(int rowIndex, int columnIndex, float value) {
		int point = points[columnIndex];
		components[point].shiftValue(rowIndex, columnIndex - splits[point], value);
	}

	@Override
	public <T extends MathMessage> ColumnCompositeMatrix collectValues(MatrixCollector<T> collector, T message, MathCalculator mode) {
		int size = components.length;
		for (int point = 0; point < size; point++) {
			MathMatrix matrix = components[point];
			int split = splits[point];
			matrix.collectValues((row, column, value, information) -> {
				collector.collect(row, column + split, value, information);
			}, message, mode);
		}
		return this;
	}

	@Override
	public <T extends MathMessage> ColumnCompositeMatrix mapValues(MatrixMapper<T> mapper, T message, MathCalculator mode) {
		int size = components.length;
		for (int point = 0; point < size; point++) {
			MathMatrix matrix = components[point];
			int split = splits[point];
			matrix.mapValues((row, column, value, information) -> {
				return mapper.map(row, column + split, value, information);
			}, message, mode);
		}
		return this;
	}

	@Override
	public MathMatrix addMatrix(MathMatrix matrix, boolean transpose) {
		for (int index = 0, size = getColumnSize(); index < size; index++) {
			getColumnVector(index).addVector(transpose ? matrix.getRowVector(index) : matrix.getColumnVector(index));
		}
		return this;
	}

	@Override
	public MathMatrix subtractMatrix(MathMatrix matrix, boolean transpose) {
		for (int index = 0, size = getColumnSize(); index < size; index++) {
			getColumnVector(index).subtractVector(transpose ? matrix.getRowVector(index) : matrix.getColumnVector(index));
		}
		return this;
	}

	@Override
	public MathMatrix multiplyMatrix(MathMatrix matrix, boolean transpose) {
		for (int index = 0, size = getColumnSize(); index < size; index++) {
			getColumnVector(index).multiplyVector(transpose ? matrix.getRowVector(index) : matrix.getColumnVector(index));
		}
		return this;
	}

	@Override
	public MathMatrix divideMatrix(MathMatrix matrix, boolean transpose) {
		for (int index = 0, size = getColumnSize(); index < size; index++) {
			getColumnVector(index).divideVector(transpose ? matrix.getRowVector(index) : matrix.getColumnVector(index));
		}
		return this;
	}

	@Override
	public MathMatrix copyMatrix(MathMatrix matrix, boolean transpose) {
		for (int index = 0, size = getColumnSize(); index < size; index++) {
			getColumnVector(index).copyVector(transpose ? matrix.getRowVector(index) : matrix.getColumnVector(index));
		}
		return this;
	}

	@Override
	public Iterator<MatrixScalar> iterator() {
		return new ColumnCompositeMatrixIterator();
	}

	private class ColumnCompositeMatrixIterator implements Iterator<MatrixScalar> {

		private int index;

		private int current = components[index].getElementSize();

		private int cursor;

		private int size = elementSize;

		private Iterator<MatrixScalar> iterator = components[index].iterator();

		private CompositeMatrixScalar term = new CompositeMatrixScalar();

		@Override
		public boolean hasNext() {
			return cursor < size;
		}

		@Override
		public MatrixScalar next() {
			if (cursor++ < current) {
				MatrixScalar scalar = iterator.next();
				term.update(scalar, scalar.getRow(), scalar.getColumn() + splits[index]);
			}
			if (cursor == current && current != size) {
				current += components[++index].getElementSize();
				iterator = components[index].iterator();
			}
			return term;
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}

	}

	private static ColumnCompositeMatrix valueOf(MathMatrix... components) {
		assert components.length != 0;
		ColumnCompositeMatrix instance = new ColumnCompositeMatrix();
		instance.components = components;
		instance.rowSize = components[0].getRowSize();
		instance.splits = new int[components.length + 1];
		instance.indexed = true;
		for (int point = 0, size = components.length; point < size; point++) {
			MathMatrix matrix = components[point];
			instance.columnSize += matrix.getColumnSize();
			instance.splits[point + 1] = instance.columnSize;
			instance.elementSize += matrix.getElementSize();
			instance.knownSize += matrix.getKnownSize();
			instance.unknownSize += matrix.getUnknownSize();
			if (instance.indexed && !matrix.isIndexed()) {
				instance.indexed = false;
			}
			matrix.attachMonitor(instance);
		}
		instance.points = new int[instance.splits[components.length]];
		int cursor = 0;
		int point = 0;
		for (MathMatrix matrix : components) {
			int size = matrix.getColumnSize();
			for (int position = 0; position < size; position++) {
				instance.points[cursor++] = point;
			}
			point++;
		}
		return instance;
	}

	public static ColumnCompositeMatrix attachOf(MathMatrix... components) {
		Collection<MathMatrix> elements = new LinkedList<>();
		for (MathMatrix component : components) {
			if (component instanceof ColumnCompositeMatrix) {
				ColumnCompositeMatrix element = ColumnCompositeMatrix.class.cast(component);
				Collections.addAll(elements, element.components);
			} else {
				elements.add(component);
			}
		}
		MathMatrix[] matrixes = elements.toArray(new MathMatrix[elements.size()]);
		return valueOf(matrixes);
	}

	public static ColumnCompositeMatrix detachOf(ColumnCompositeMatrix components, int from, int to) {
		Collection<MathMatrix> elements = new LinkedList<>();
		for (int point = 0, size = components.splits.length; point < size; point++) {
			int split = components.splits[point];
			if (split == to) {
				break;
			}
			if (split == from) {
				elements.add(components.components[point]);
			} else if (!elements.isEmpty()) {
				elements.add(components.components[point]);
			}
		}
		MathMatrix[] matrixes = elements.toArray(new MathMatrix[elements.size()]);
		return valueOf(matrixes);
	}

}
