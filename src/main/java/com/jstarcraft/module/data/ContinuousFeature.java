package com.jstarcraft.module.data;

import java.util.Iterator;
import java.util.LinkedList;

public class ContinuousFeature implements DataFeature<Float> {

	public final static int DEFAULT_CAPACITY = 10000;

	private ContinuousAttribute attribute;

	/** 容量 */
	private int capacity;

	/** 数组 */
	private float[] current;

	/** 特征名 */
	private String name;

	/** 特征值 */
	private LinkedList<float[]> values;

	/** 大小 */
	private int size;

	public ContinuousFeature(String name, ContinuousAttribute attribute) {
		this.attribute = attribute;
		this.capacity = DEFAULT_CAPACITY;
		this.name = name;
		this.values = new LinkedList<>();
		this.size = 0;
	}

	@Override
	public void associate(Object data) {
		int position = size++ % capacity;
		if (position == 0) {
			current = new float[capacity];
			values.add(current);
		}
		current[position] = attribute.makeValue(data);
	}

	@Override
	public ContinuousAttribute getAttribute() {
		return attribute;
	}

	@Override
	public String getName() {
		return name;
	}

	@Override
	public int getSize() {
		return size;
	}

	@Override
	public Iterator<Float> iterator() {
		return new ContinuousFeatureIterator();
	}

	private class ContinuousFeatureIterator implements Iterator<Float> {

		private int cursor = 0;

		private float[] current;

		private final Iterator<float[]> iterator = values.iterator();

		@Override
		public boolean hasNext() {
			return cursor < size;
		}

		@Override
		public Float next() {
			int position = cursor++ % capacity;
			if (position == 0) {
				current = iterator.next();
			}
			return current[position];
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}

	}

}
