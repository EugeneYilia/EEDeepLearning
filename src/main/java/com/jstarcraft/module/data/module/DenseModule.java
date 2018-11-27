package com.jstarcraft.module.data.module;

import com.jstarcraft.module.data.DataInstance;
import com.jstarcraft.module.data.DataModule;
import com.jstarcraft.module.data.FloatArray;
import com.jstarcraft.module.data.IntegerArray;

import it.unimi.dsi.fastutil.ints.Int2FloatMap;
import it.unimi.dsi.fastutil.ints.Int2IntMap;

/**
 * 稠密模块
 * 
 * @author Birdy
 *
 */
public class DenseModule implements DataModule {

	/** 离散特征 */
	private int discreteDimension;

	private IntegerArray[] discreteValues;

	/** 连续特征 */
	private int continuousDimension;

	private FloatArray[] continuousValues;

	public DenseModule(int discreteDimension, int continuousDimension, int instanceCapacity) {
		this.discreteDimension = discreteDimension;
		this.discreteValues = new IntegerArray[discreteDimension];
		for (int index = 0; index < discreteDimension; index++) {
			this.discreteValues[index] = new IntegerArray(1000, instanceCapacity);
		}
		this.continuousDimension = continuousDimension;
		for (int index = 0; index < continuousDimension; index++) {
			this.continuousValues[index] = new FloatArray(1000, instanceCapacity);
		}
	}

	int getDiscreteDimension() {
		return discreteDimension;
	}

	IntegerArray[] getDiscreteValues() {
		return discreteValues;
	}

	int getContinuousDimension() {
		return continuousDimension;
	}

	FloatArray[] getContinuousValues() {
		return continuousValues;
	}

	@Override
	public void associateInstance(Int2IntMap discreteFeatures, Int2FloatMap continuousFeatures) {
		assert discreteValues.length == discreteFeatures.size();
		assert continuousValues.length == continuousFeatures.size();
		for (Int2IntMap.Entry term : discreteFeatures.int2IntEntrySet()) {
			discreteValues[term.getIntKey()].associateData(term.getIntValue());
		}
		for (Int2FloatMap.Entry term : continuousFeatures.int2FloatEntrySet()) {
			continuousValues[term.getIntKey()].associateData(term.getFloatValue());
		}
	}

	@Override
	public DataInstance getInstance(int cursor) {
		return new DenseInstance(cursor, this);
	}

	@Override
	public int getSize() {
		// TODO Auto-generated method stub
		return 0;
	}

}
