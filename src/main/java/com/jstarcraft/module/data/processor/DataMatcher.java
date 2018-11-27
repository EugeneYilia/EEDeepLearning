package com.jstarcraft.module.data.processor;

import com.jstarcraft.module.data.DataAccessor;
import com.jstarcraft.module.data.DiscreteAttribute;

/**
 * 数据匹配器
 * 
 * @author Birdy
 *
 */
public interface DataMatcher {

	public static DataMatcher discreteOf(DataAccessor<?> accessor, int dimension) {
		DiscreteAttribute attribute = accessor.getDiscreteAttribute(dimension);
		int size = accessor.getSize();
		return (paginations, positions) -> {
			if (paginations.length != attribute.getSize() + 1) {
				throw new IllegalArgumentException();
			}
			if (positions.length != size) {
				throw new IllegalArgumentException();
			}
			for (int index = 0; index < size; index++) {
				int feature = accessor.getDiscreteFeature(dimension, index);
				paginations[feature + 1]++;
			}
			int cursor = size;
			for (int index = paginations.length - 1; index > 0; index--) {
				cursor -= paginations[index];
				paginations[index] = cursor;
			}
			for (int index = 0; index < size; index++) {
				int feature = accessor.getDiscreteFeature(dimension, index);
				positions[paginations[feature + 1]++] = index;
			}
		};
	}

	void match(int[] paginations, int[] positions);

}
