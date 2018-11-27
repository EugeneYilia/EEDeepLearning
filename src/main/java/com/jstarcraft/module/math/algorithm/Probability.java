package com.jstarcraft.module.math.algorithm;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.math.structure.vector.SparseVector;
import com.jstarcraft.module.math.structure.vector.VectorMapper;
import com.jstarcraft.module.recommendation.exception.RecommendationException;

/**
 * 随机概率
 * 
 * @author Birdy
 *
 */
// TODO 实质为概率质量函数(Probability Mass Function)
// TODO 准备改名为SampleProbability,WeightProbability,RandomProbability或者SpecificProbability
public class Probability {

	/** 中位值 */
	private float median;

	/** 累计值 */
	private float sum;

	/** 概率值 */
	private float[] values;

	public Probability(int size, VectorMapper<?> accessor) {
		values = new float[size];
		sum = 0F;
		for (int index = 0; index < size; index++) {
			sum += accessor.map(index, values[index], null);
			values[index] = sum;
		}
		median = values[values.length / 2];
	}

	public Probability calculate(VectorMapper<?> accessor) {
		sum = 0F;
		float value = 0F;
		for (int index = 0, size = values.length; index < size; index++) {
			sum += accessor.map(index, values[index] - value, null);
			value = values[index];
			values[index] = sum;
		}
		median = values[values.length / 2];
		return this;
	}

	public int random() {
		double random = RandomUtility.randomDouble(sum);
		for (int index = random < median ? 0 : values.length / 2, size = values.length; index < size; index++) {
			if (values[index] >= random) {
				return index;
			}
		}
		throw new RecommendationException("概率范围超过随机范围,检查是否由于多线程修改导致.");
	}

	/**
	 * TODO 此算法比较慢,需要考虑优化.
	 * 
	 * @param vector
	 * @return
	 */
	public Integer random(SparseVector vector) {
		double random = RandomUtility.randomDouble(sum);
		int position = 0;
		int index;
		int size = vector.getElementSize();
		while (position < size) {
			index = vector.getIndex(position++);
			double from = index == 0 ? 0D : values[index - 1];
			double to = values[index];
			if (random >= from && random < to) {
				return null;
			}
		}

		for (index = random < median ? 0 : values.length / 2, size = values.length; index < size; index++) {
			if (values[index] >= random) {
				return index;
			}
		}
		throw new RecommendationException("概率范围超过随机范围,检查是否由于多线程修改导致.");
	}

	public double getSum() {
		return sum;
	}

}
