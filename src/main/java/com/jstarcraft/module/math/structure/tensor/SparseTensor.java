package com.jstarcraft.module.math.structure.tensor;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.TreeMap;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.MathAccessor;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.MathIterator;
import com.jstarcraft.module.math.structure.MathMessage;
import com.jstarcraft.module.math.structure.matrix.MatrixScalar;
import com.jstarcraft.module.math.structure.matrix.SparseMatrix;

/**
 * 稀疏张量
 * 
 * <pre>
 * TODO 张量永远为N个离散特征+1个连续特征.
 * 数据装载到内存中,都会以张量的形式.划分为训练张量,测试张量以及上下文张量.
 * 
 * 推荐器根据需要再将张量转换为各种矩阵与向量.
 * 
 * 张量概念解释:
 * http://wiki.jikexueyuan.com/project/tensorflow-zh/resources/dims_types.html
 * </pre>
 * 
 * @author Birdy
 *
 */
public class SparseTensor implements MathTensor {

	public final static int DEFAULT_CAPACITY = 10000;

	/** 维度 */
	private int[] dimensions;

	/** 索引 */
	private int[][] indexes;

	/** 值 */
	private float[] values;

	private SparseTensor() {
	}

	@Override
	public int getElementSize() {
		return values.length;
	}

	@Override
	public int getKnownSize() {
		return getElementSize();
	}

	@Override
	public int getUnknownSize() {
		int total = 0;
		for (int size : dimensions) {
			total += size;
		}
		return total - getElementSize();
	}

	@Override
	public MathIterator<TensorScalar> iterateElement(MathCalculator mode, MathAccessor<TensorScalar>... accessors) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getOrderSize() {
		return dimensions.length;
	}

	@Override
	public int getDimensionSize(int dimension) {
		return dimensions[dimension];
	}

	@Override
	public int getIndex(int dimension, int position) {
		return indexes[dimension][position];
	}

	@Override
	public float getValue(int position) {
		return values[position];
	}

	@Override
	public <T extends MathMessage> SparseTensor calculate(TensorCollector<T> collector, T message, MathCalculator mode) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public <T extends MathMessage> SparseTensor calculate(TensorMapper<T> mapper, T message, MathCalculator mode) {
		// TODO Auto-generated method stub
		return null;
	}

	// /**
	// * 分组数据实例
	// *
	// * @param dimension
	// * @return
	// */
	// public int[][][] groupIndexes(int dimension) {
	// int size = dimensionSizes[dimension];
	// int[] counts = new int[size];
	// int[] sizes = new int[size];
	// int[][][] groups = new int[size][][];
	// for (int[] instance : indexes) {
	// sizes[instance[dimension]]++;
	// }
	// for (int[] instance : indexes) {
	// int index = instance[dimension];
	// int[][] group = groups[index];
	// if (group == null) {
	// group = new int[sizes[index]][];
	// groups[instance[dimension]] = group;
	// }
	// group[counts[index]++] = instance;
	// }
	// return groups;
	// }
	//
	// /**
	// * 排序数据实例
	// *
	// * @param comparator
	// * @return
	// */
	// public int[][] sortIndexes(IndexSorter sorter) {
	// int[][] sorts = new int[indexes.length][];
	// int index = 0;
	// for (int[] instance : indexes) {
	// sorts[index++] = instance;
	// }
	// sorter.sort(sorts);
	// return sorts;
	// }

	/**
	 * retrieve a rating matrix from the tensor. Warning: it assumes there is at
	 * most one entry for each (user, item) pair.
	 *
	 * @return a sparse rating matrix
	 */
	// TODO 考虑取消(整合到SparseMatrix)
	@Deprecated
	public SparseMatrix toMatrix(int rowDimension, int columnDimension) {
		Table<Integer, Integer, Float> dataTable = HashBasedTable.create();
		int size = values.length;
		for (int position = 0; position < size; position++) {
			int rowIndex = indexes[rowDimension][position];
			int columnIndex = indexes[columnDimension][position];
			// TODO 处理冲突
			dataTable.put(rowIndex, columnIndex, values[position]);
		}
		return SparseMatrix.valueOf(dimensions[rowDimension], dimensions[columnDimension], dataTable);
	}

	@Override
	public Iterator<TensorScalar> iterator() {
		return new SparseTensorIterator(indexes, values);
	}

	public static <T> SparseTensor valueOf(int[] dimensions, Iterable<T> iterator, TensorTransformer<T> transformer) {
		List<int[][]> globalIndexes = new ArrayList<>(100);
		List<float[]> globalValues = new ArrayList<>(100);
		// 保证排序与唯一
		TreeMap<Integer, Integer> positions = new TreeMap<>((left, right) -> {
			int[][] leftIndexes = globalIndexes.get(left / DEFAULT_CAPACITY);
			int[][] rightIndexes = globalIndexes.get(right / DEFAULT_CAPACITY);
			int value = 0;
			for (int dimension = 0; dimension < dimensions.length; dimension++) {
				value = leftIndexes[dimension][left % DEFAULT_CAPACITY] - rightIndexes[dimension][right % DEFAULT_CAPACITY];
				if (value != 0) {
					break;
				}
			}
			return value;
		});
		int[][] localIndexes = new int[dimensions.length][DEFAULT_CAPACITY];
		float[] localValues = new float[DEFAULT_CAPACITY];
		globalIndexes.add(localIndexes);
		globalValues.add(localValues);
		KeyValue<int[], Float> keyValue = new KeyValue<>(new int[dimensions.length], null);
		int newPosition = 0;
		for (T instance : iterator) {
			int cursor = newPosition % DEFAULT_CAPACITY;
			transformer.transform(keyValue, instance);
			for (int dimension = 0; dimension < dimensions.length; dimension++) {
				localIndexes[dimension][cursor] = keyValue.getKey()[dimension];
			}
			localValues[cursor] = keyValue.getValue();

			Integer oldPosition = positions.get(newPosition);
			if (oldPosition == null) {
				positions.put(newPosition, newPosition);
				newPosition++;
				if (newPosition % DEFAULT_CAPACITY == 0) {
					localIndexes = new int[dimensions.length][DEFAULT_CAPACITY];
					globalIndexes.add(localIndexes);
					localValues = new float[DEFAULT_CAPACITY];
					globalValues.add(localValues);
				}
			} else {
				// TODO 处理冲突
				System.out.println("覆盖数据" + oldPosition + " " + newPosition);
			}
		}

		SparseTensor tensor = new SparseTensor();
		tensor.dimensions = dimensions;
		tensor.indexes = new int[dimensions.length][newPosition];
		tensor.values = new float[newPosition];
		newPosition = 0;
		for (int oldPosition : positions.keySet()) {
			int cursor = oldPosition % DEFAULT_CAPACITY;
			localIndexes = globalIndexes.get(oldPosition / DEFAULT_CAPACITY);
			localValues = globalValues.get(oldPosition / DEFAULT_CAPACITY);
			for (int order = 0; order < dimensions.length; order++) {
				tensor.indexes[order][newPosition] = localIndexes[order][cursor];
			}
			tensor.values[newPosition] = localValues[cursor];
			newPosition++;
		}
		return tensor;
	}

	@Override
	public MathIterator<TensorScalar> setValues(float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public MathIterator<TensorScalar> scaleValues(float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public MathIterator<TensorScalar> shiftValues(float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public float getSum(boolean absolute) {
		// TODO Auto-generated method stub
		return 0F;
	}

}

class SparseTensorIterator implements Iterator<TensorScalar> {

	private int cursor;

	private int size;

	private SparseTensorScalar term;

	SparseTensorIterator(int[][] indexes, float[] values) {
		this.size = values.length;
		this.term = new SparseTensorScalar(0, indexes, values);
	}

	public boolean hasNext() {
		return cursor < size;
	}

	public TensorScalar next() {
		return term.update(cursor++);
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

}

class SparseTensorScalar implements TensorScalar {

	private int[][] indexes;

	private float[] values;

	private int cursor;

	public SparseTensorScalar(int cursor, int[][] indexes, float[] values) {
		this.cursor = cursor;
		this.indexes = indexes;
		this.values = values;
	}

	SparseTensorScalar update(int cursor) {
		this.cursor = cursor;
		return this;
	}

	@Override
	public int getIndex(int dimension) {
		return indexes[dimension][cursor];
	}

	@Override
	public int[] getIndexes(int[] indexes) {
		for (int dimension = 0; dimension < this.indexes.length; dimension++) {
			indexes[dimension] = this.indexes[dimension][cursor];
		}
		return indexes;
	}

	@Override
	public float getValue() {
		return values[cursor];
	}

	@Override
	public void scaleValue(float value) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void setValue(float value) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void shiftValue(float value) {
		throw new UnsupportedOperationException();
	}

}
