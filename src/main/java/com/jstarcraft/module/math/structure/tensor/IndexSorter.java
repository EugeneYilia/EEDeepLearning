package com.jstarcraft.module.math.structure.tensor;

import java.util.Arrays;

import com.jstarcraft.core.utility.RandomUtility;

/**
 * 数据排序器
 * 
 * @author Birdy
 *
 */
public interface IndexSorter {

	/**
	 * 排序指定的数据实例列表
	 * 
	 * @param instances
	 */
	void sort(int[][] instances);

	public final static IndexSorter RANDOM_SORTER = (instances) -> {
		RandomUtility.shuffle(instances);
	};

	public static IndexSorter getDimensionSorter(int dimension) {
		return (instances) -> {
			Arrays.sort(instances, (left, right) -> {
				// TODO 注意:此处存在0的情况.
				return left[dimension] - right[dimension];
			});
		};
	}

}
