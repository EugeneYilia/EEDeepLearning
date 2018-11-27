package com.jstarcraft.module.neuralnetwork.schedule;

/**
 * ISchedule: a general purpose interface for getting values according to some
 * schedule. Used for implementing learning rate, dropout and momentum schedules
 * - and in principle, any univariate (double) value that deponds on the current
 * iteration and epochs numbers.<br>
 * <br>
 * Note: ISchedule objects should not have mutable state - i.e., they should be
 * safe to share between multiple locations/layers.
 *
 * @author Alex Black
 */
public interface Schedule {

	/**
	 * @param iteration
	 *            Current iteration number. Starts at 0
	 * @param epoch
	 *            Current epoch number. Starts at 0
	 * @return Value at the current iteration/epoch for this schedule
	 */
	float valueAt(int iteration, int epoch);

}
