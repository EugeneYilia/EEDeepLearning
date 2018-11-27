package com.jstarcraft.module.data.module;

import com.jstarcraft.module.data.DataInstance;
import com.jstarcraft.module.data.DataModule;
import com.jstarcraft.module.data.FloatArray;
import com.jstarcraft.module.data.IntegerArray;

import it.unimi.dsi.fastutil.ints.Int2FloatMap;
import it.unimi.dsi.fastutil.ints.Int2IntMap;

/**
 * 稀疏模块
 * 
 * @author Birdy
 *
 */
public class SparseModule implements DataModule {

	/** 离散特征 */
	private int discreteDimension;

	private IntegerArray discretePoints;

	private IntegerArray discreteIndexes;

	private IntegerArray discreteValues;

	/** 连续特征 */
	private int continuousDimension;

	private IntegerArray continuousPoints;

	private IntegerArray continuousIndexes;

	private FloatArray continuousValues;

	public SparseModule(int discreteDimension, int continuousDimension, int instanceCapacity) {
		this.discreteDimension = discreteDimension;
		this.discretePoints = new IntegerArray(1000, instanceCapacity + 1);
		this.discreteIndexes = new IntegerArray(1000, instanceCapacity * discreteDimension);
		this.discreteValues = new IntegerArray(1000, instanceCapacity * discreteDimension);
		this.continuousDimension = continuousDimension;
		this.continuousPoints = new IntegerArray(1000, instanceCapacity + 1);
		this.continuousIndexes = new IntegerArray(1000, instanceCapacity * continuousDimension);
		this.continuousValues = new FloatArray(1000, instanceCapacity * continuousDimension);
		this.discretePoints.associateData(0);
		this.continuousPoints.associateData(0);
	}

	int getDiscreteDimensions() {
		return discreteDimension;
	}

	IntegerArray getDiscretePoints() {
		return discretePoints;
	}

	IntegerArray getDiscreteIndexes() {
		return discreteIndexes;
	}

	IntegerArray getDiscreteValues() {
		return discreteValues;
	}

	int getContinuousDimensions() {
		return continuousDimension;
	}

	IntegerArray getContinuousPoints() {
		return continuousPoints;
	}

	IntegerArray getContinuousIndexes() {
		return continuousIndexes;
	}

	FloatArray getContinuousValues() {
		return continuousValues;
	}

	@Override
	public void associateInstance(Int2IntMap discreteFeatures, Int2FloatMap continuousFeatures) {
		discretePoints.associateData(discretePoints.getData(discretePoints.getSize() - 1) + discreteFeatures.size());
		continuousPoints.associateData(continuousPoints.getData(continuousPoints.getSize() - 1) + continuousFeatures.size());
		for (Int2IntMap.Entry term : discreteFeatures.int2IntEntrySet()) {
			discreteIndexes.associateData(term.getIntKey());
			discreteValues.associateData(term.getIntValue());
		}
		for (Int2FloatMap.Entry term : continuousFeatures.int2FloatEntrySet()) {
			continuousIndexes.associateData(term.getIntKey());
			continuousValues.associateData(term.getFloatValue());
		}
	}

	@Override
	public DataInstance getInstance(int cursor) {
		return new SparseInstance(cursor, this);
	}

	@Override
	public int getSize() {
		// TODO Auto-generated method stub
		return 0;
	}

}
