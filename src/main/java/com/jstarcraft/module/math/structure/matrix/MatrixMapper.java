package com.jstarcraft.module.math.structure.matrix;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.math.algorithm.distribution.ContinuousProbability;
import com.jstarcraft.module.math.algorithm.distribution.DiscreteProbability;
import com.jstarcraft.module.math.structure.MathMessage;

/**
 * 矩阵映射器(会导致数据变化)
 * 
 * <pre>
 * 与{@link MathMatrix}的calculate方法相关.
 * 例如:重置,随机.
 * </pre>
 * 
 * @author Birdy
 *
 */
@Deprecated
public interface MatrixMapper<T extends MathMessage> {

	public static final MatrixMapper<?> ZERO = constantOf(0F);

	public static final MatrixMapper<?> RANDOM = randomOf(1F);

	@Deprecated
	public static MatrixMapper<?> constantOf(float constant) {
		return (row, column, value, message) -> {
			return constant;
		};
	}

	@Deprecated
	public static MatrixMapper<?> copyOf(MathMatrix matrix) {
		assert matrix.isIndexed();
		return (row, column, value, message) -> {
			return matrix.getValue(row, column);
		};
	}

	public static MatrixMapper<?> distributionOf(ContinuousProbability probability) {
		return (row, column, value, message) -> {
			return probability.sample().floatValue();
		};
	}

	public static MatrixMapper<?> distributionOf(DiscreteProbability probability) {
		return (row, column, value, message) -> {
			return probability.sample();
		};
	}

	public static MatrixMapper<?> randomOf(float random) {
		return (row, column, value, message) -> {
			return RandomUtility.randomFloat(random);
		};
	}

	public static MatrixMapper<?> randomOf(int random) {
		return (row, column, value, message) -> {
			return RandomUtility.randomInteger(random);
		};
	}

	float map(int row, int column, float value, T message);

}
