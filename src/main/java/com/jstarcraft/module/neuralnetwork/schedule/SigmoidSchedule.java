package com.jstarcraft.module.neuralnetwork.schedule;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

/**
 * Sigmoid decay schedule, with 3 parameters: initial value, gamma and
 * stepSize.<br>
 * value(i) = initialValue * 1.0 / (1 + exp(-gamma * (iter - stepSize))) where i
 * is the iteration or epoch (depending on the setting)
 *
 * @author Alex Black
 */
public class SigmoidSchedule implements Schedule {

	private ScheduleType scheduleType;
	private float initialValue;
	private float gamma;
	private int stepSize;

	SigmoidSchedule() {
	}

	public SigmoidSchedule(ScheduleType scheduleType, float initialValue, float gamma, int stepSize) {
		this.scheduleType = scheduleType;
		this.initialValue = initialValue;
		this.gamma = gamma;
		this.stepSize = stepSize;
	}

	@Override
	public float valueAt(int iteration, int epoch) {
		int i = (scheduleType == ScheduleType.ITERATION ? iteration : epoch);
		return (float) (initialValue / (1F + Math.exp(-gamma * (i - stepSize))));
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
			SigmoidSchedule that = (SigmoidSchedule) object;
			EqualsBuilder equal = new EqualsBuilder();
			equal.append(this.scheduleType, that.scheduleType);
			equal.append(this.initialValue, that.initialValue);
			equal.append(this.gamma, that.gamma);
			equal.append(this.stepSize, that.stepSize);
			return equal.isEquals();
		}
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(scheduleType);
		hash.append(initialValue);
		hash.append(gamma);
		hash.append(stepSize);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "SigmoidSchedule(scheduleType=" + scheduleType + ", initialValue=" + initialValue + ", gamma=" + gamma + ", stepSize=" + stepSize + ")";
	}

}
