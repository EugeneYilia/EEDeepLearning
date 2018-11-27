package com.jstarcraft.module.data.module;

import com.jstarcraft.module.data.DataInstance;
import com.jstarcraft.module.data.FloatArray;
import com.jstarcraft.module.data.IntegerArray;

public class SparseInstance implements DataInstance {

	private int cursor;

	/** 离散特征 */
	private int[] discreteFeatures;

	private IntegerArray discretePoints;

	private IntegerArray discreteIndexes;

	private IntegerArray discreteValues;

	/** 连续特征 */
	private float[] continuousFeatures;

	private IntegerArray continuousPoints;

	private IntegerArray continuousIndexes;

	private FloatArray continuousValues;

	SparseInstance(int cursor, SparseModule module) {
		this.cursor = cursor;
		this.discreteFeatures = new int[module.getDiscreteDimensions()];
		{
			for (int index = 0, size = module.getDiscreteDimensions(); index < size; index++) {
				this.discreteFeatures[index] = -1;
			}
		}
		this.discretePoints = module.getDiscretePoints();
		this.discreteIndexes = module.getDiscreteIndexes();
		this.discreteValues = module.getDiscreteValues();
		this.continuousFeatures = new float[module.getContinuousDimensions()];
		{
			for (int index = 0, size = module.getContinuousDimensions(); index < size; index++) {
				this.continuousFeatures[index] = Float.NaN;
			}
		}
		this.continuousPoints = module.getContinuousPoints();
		this.continuousIndexes = module.getContinuousIndexes();
		this.continuousValues = module.getContinuousValues();
		{
			int from = this.discretePoints.getData(this.cursor);
			int to = this.discretePoints.getData(this.cursor + 1);
			for (int position = from; position < to; position++) {
				int index = this.discreteIndexes.getData(position);
				this.discreteFeatures[index] = this.discreteValues.getData(position);
			}
		}
		{
			int from = this.continuousPoints.getData(this.cursor);
			int to = this.continuousPoints.getData(this.cursor + 1);
			for (int position = from; position < to; position++) {
				int index = this.continuousIndexes.getData(position);
				this.continuousFeatures[index] = this.continuousValues.getData(position);
			}
		}
	}

	@Override
	public void setCursor(int cursor) {
		{
			int from = this.discretePoints.getData(this.cursor);
			int to = this.discretePoints.getData(this.cursor + 1);
			for (int position = from; position < to; position++) {
				int index = this.discreteIndexes.getData(position);
				this.discreteFeatures[index] = -1;
			}
		}
		{
			int from = this.continuousPoints.getData(this.cursor);
			int to = this.continuousPoints.getData(this.cursor + 1);
			for (int position = from; position < to; position++) {
				int index = this.continuousIndexes.getData(position);
				this.continuousFeatures[index] = Float.NaN;
			}
		}
		this.cursor = cursor;
		{
			int from = this.discretePoints.getData(this.cursor);
			int to = this.discretePoints.getData(this.cursor + 1);
			for (int position = from; position < to; position++) {
				int index = this.discreteIndexes.getData(position);
				this.discreteFeatures[index] = this.discreteValues.getData(position);
			}
		}
		{
			int from = this.continuousPoints.getData(this.cursor);
			int to = this.continuousPoints.getData(this.cursor + 1);
			for (int position = from; position < to; position++) {
				int index = this.continuousIndexes.getData(position);
				this.continuousFeatures[index] = this.continuousValues.getData(position);
			}
		}
	}

	@Override
	public int getCursor() {
		return cursor;
	}

	@Override
	public int getDiscreteFeature(int index) {
		return this.discreteFeatures[index];
	}

	@Override
	public float getContinuousFeature(int index) {
		return this.continuousFeatures[index];
	}

}
