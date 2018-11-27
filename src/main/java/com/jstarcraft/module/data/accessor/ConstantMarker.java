package com.jstarcraft.module.data.accessor;

import java.util.LinkedHashMap;

import com.jstarcraft.module.data.ContinuousAttribute;
import com.jstarcraft.module.data.DiscreteAttribute;
import com.jstarcraft.module.data.IntegerArray;

/**
 * 常量标记器
 * 
 * @author Birdy
 *
 */
public class ConstantMarker extends SampleAccessor {

	private float constant;

	public ConstantMarker(IntegerArray positions, InstanceAccessor model, float constant) {
		this.discreteAttributes = new DiscreteAttribute[model.discreteAttributes.length];
		this.continuousAttributes = new ContinuousAttribute[model.continuousAttributes.length];
		this.discreteFeatures = new int[model.discreteAttributes.length][];
		this.continuousFeatures = new float[model.continuousAttributes.length][];
		for (int index = 0, size = model.discreteAttributes.length; index < size; index++) {
			this.discreteAttributes[index] = model.discreteAttributes[index];
			this.discreteFeatures[index] = model.discreteFeatures[index];
		}
		for (int index = 0, size = model.continuousAttributes.length; index < size; index++) {
			this.continuousAttributes[index++] = model.continuousAttributes[index];
			this.continuousFeatures[index++] = model.continuousFeatures[index];
		}
		this.discreteDimensions = new LinkedHashMap<>();
		for (String field : model.getDiscreteFields()) {
			this.discreteDimensions.put(field, model.getDiscreteDimension(field));
		}
		this.continuousDimensions = new LinkedHashMap<>();
		for (String field : model.getContinuousFields()) {
			this.continuousDimensions.put(field, model.getContinuousDimension(field));
		}
		this.positions = positions;
		this.constant = constant;
	}

	@Override
	public float getMark(int position) {
		return constant;
	}

}
