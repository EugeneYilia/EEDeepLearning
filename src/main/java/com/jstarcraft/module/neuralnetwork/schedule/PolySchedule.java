package com.jstarcraft.module.neuralnetwork.schedule;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

/**
 *
 * Polynomial decay schedule, with 3 parameters: initial value, maxIter,
 * power.<br>
 * Note that the the value will be 0 after maxIter, otherwise is given by:
 * value(i) = initialValue * (1 + i/maxIter)^(-power) where i is the iteration
 * or epoch (depending on the setting)
 *
 * @author Alex Black
 */
public class PolySchedule implements Schedule {

	private ScheduleType scheduleType;
	private float initialValue;
	private float power;
	private int maxIter;

	PolySchedule() {
	}

	public PolySchedule(ScheduleType scheduleType, float initialValue, float power, int maxIter) {
		this.scheduleType = scheduleType;
		this.initialValue = initialValue;
		this.power = power;
		this.maxIter = maxIter;
	}

	@Override
	public float valueAt(int iteration, int epoch) {
		int i = (scheduleType == ScheduleType.ITERATION ? iteration : epoch);

		if (i >= maxIter) {
			return 0;
		}

		return (float) (initialValue * Math.pow(1D + i / (float) maxIter, power));
	}

	@Override
	public boolean equals(Object object) {
		if (this == object) {
			return true;
		}
		if (object == null) {
			return false;
		}
		if (getClass() != object.getClass()) {
			return false;
		} else {
			PolySchedule that = (PolySchedule) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.scheduleType, that.scheduleType);
			equal.append(this.initialValue, that.initialValue);
			equal.append(this.power, that.power);
			equal.append(this.maxIter, that.maxIter);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(scheduleType);
		hash.append(initialValue);
		hash.append(power);
		hash.append(maxIter);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "PolySchedule(scheduleType=" + scheduleType + ", initialValue=" + initialValue + ", power=" + power + ", maxIter=" + maxIter + ")";
	}

}
