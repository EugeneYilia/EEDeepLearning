package com.jstarcraft.module.math.structure.vector;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.math.algorithm.distribution.ContinuousProbability;
import com.jstarcraft.module.math.algorithm.distribution.DiscreteProbability;
import com.jstarcraft.module.math.structure.MathMessage;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;

/**
 * 向量映射器(会导致数据变化)
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
public interface VectorMapper<T extends MathMessage> {

	public static final VectorMapper<?> ZERO = constantOf(0F);

	public static final VectorMapper<?> RANDOM = randomOf(1F);

	@Deprecated
	public static VectorMapper<?> constantOf(float constant) {
		return (index, value, message) -> {
			return constant;
		};
	}

	@Deprecated
	public static VectorMapper<?> copyOf(MathVector vector) {
		return (index, value, message) -> {
			return vector.getValue(index);
		};
	}

	public static VectorMapper<?> distributionOf(ContinuousProbability probability) {
		return (index, value, message) -> {
			return probability.sample().floatValue();
		};
	}

	public static VectorMapper<?> distributionOf(DiscreteProbability probability) {
		return (index, value, message) -> {
			return probability.sample();
		};
	}

	public static VectorMapper<?> randomOf(float random) {
		return (index, value, message) -> {
			return RandomUtility.randomFloat(random);
		};
	}

	float map(int index, float value, T message);

}
