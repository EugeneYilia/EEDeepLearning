package com.jstarcraft.module.data.module;

import com.jstarcraft.module.data.DataInstance;
import com.jstarcraft.module.data.FloatArray;
import com.jstarcraft.module.data.IntegerArray;

public class DenseInstance implements DataInstance {

	private int cursor;

	/** 离散特征 */
	private IntegerArray[] discreteValues;

	/** 连续特征 */
	private FloatArray[] continuousValues;

	DenseInstance(int cursor, DenseModule module) {
		this.cursor = cursor;
		this.discreteValues = module.getDiscreteValues();
		this.continuousValues = module.getContinuousValues();
	}

	@Override
	public void setCursor(int cursor) {
		this.cursor = cursor;
	}

	@Override
	public int getCursor() {
		return cursor;
	}

	@Override
	public int getDiscreteFeature(int index) {
		return discreteValues[index].getData(cursor);
	}

	@Override
	public float getContinuousFeature(int index) {
		return continuousValues[index].getData(cursor);
	}

}
