package com.jstarcraft.module.neuralnetwork.condition.batch;

import java.util.Map;
import java.util.concurrent.TimeUnit;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.neuralnetwork.condition.Condition;

/**
 * 时间限制条件
 */
public class TimeLimitCondition implements Condition {

	private long timeDuration;

	private TimeUnit timeUnit;

	private transient long timeStamp;

	public TimeLimitCondition(long timeDuration, TimeUnit timeUnit) {
		assert timeDuration > 0 && timeUnit != null;
		this.timeDuration = timeDuration;
		this.timeUnit = timeUnit;
	}

	@Override
	public void start() {
		this.timeStamp = TimeUnit.MILLISECONDS.convert(timeDuration, timeUnit);
	}

	@Override
	public boolean stop(double newScore, double oldScore, Map<String, MathMatrix> gradients) {
		return System.currentTimeMillis() >= timeStamp;
	}

	@Override
	public String toString() {
		return "MaxTimeIterationTerminationCondition(" + timeDuration + ",unit=" + timeUnit + ")";
	}

}
